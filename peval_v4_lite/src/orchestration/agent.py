import os
from typing import Optional, List
from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME

from ..core.state import PEVState
from ..core.config import PEVConfig
from ..core.engine import PEVEngine
from ..memory.wisdom import WisdomStore
from ..metacognition.learner import GlobalLearner
from ..core.logger import PEVLogger

class PEVALAgent(Agent):
    """
    The Official PEVAL Phase 4 Lite Agent.
    Modular, Resilient, Resource-Aware, and LangGraph-Free.
    """
    def __init__(self, tools_info: List, wiki: str):
        self.config = PEVConfig()
        self.tools_info = tools_info
        self.wiki = wiki
        
        # Initialize Engine and Persistence
        self.engine = PEVEngine(tools_info, wiki)
        self.wisdom_store = WisdomStore(self.config.WISDOM_FILENAME)
        self.global_learner = GlobalLearner()

    def solve(self, env: Env, task_index: Optional[int] = None, max_steps: int = 30) -> SolveResult:
        print(f"\n{PEVLogger.HEADER}{PEVLogger.BOLD}=========================================")
        print(f"       STARTING TAU-BENCH TASK {task_index}")
        print(f"========================================={PEVLogger.RESET}\n")
        
        # 1. Initialize State (The 11-Step Lifecycle begins)
        state = PEVState()
        state.global_wisdom = self.wisdom_store(state)["global_wisdom"]
        
        env_res = env.reset(task_index=task_index)
        state.history.append({"role": "user", "content": env_res.observation})
        
        # Safe Checkpoint (The state after the last successful tool call)
        checkpoint_state = state
        
        for i in range(max_steps):
            PEVLogger.step(i + 1)
            
            # INVOKE THE NATIVE ENGINE
            # This handles Distillation -> Planning -> Execution -> Verification
            try:
                final_state_data = self.engine.invoke(state, {"recursion_limit": self.config.RECURSION_LIMIT})
                state = PEVState(**final_state_data)
                
                # Check for Inference Stall (Timeout/Concurrency Error on Gaudi)
                if state.is_stalled:
                    PEVLogger.error("STALL DETECTED: Model timed out on HPU. Retrying turnaround privately...")
                    state = checkpoint_state
                    state.is_stalled = False
                    continue

                # Check for "Abject Failure" (Reasoning Loop)
                if state.consecutive_errors >= 3:
                    PEVLogger.error("CRITICAL STAGNATION: Rolling back to last safe checkpoint.")
                    state = checkpoint_state
                    state.audit_feedback = "INSTRUCTION UPDATE: State restored to last successful action. Please pivot your ROADMAP to avoid this redundancy."
                    state.consecutive_errors = 0
                    continue

            except Exception as e:
                PEVLogger.error(f"Unplugging Engine Exception: {e}")
                break

            # Dispatch Verified Action to Tau-Bench
            action_data = state.current_action_draft
            action_name = action_data.get("name", RESPOND_ACTION_NAME)
            action_kwargs = action_data.get("arguments", {})
            
            if not action_name:
                PEVLogger.warn("Empty action drafted. Rolling back.")
                state = checkpoint_state
                continue
                
            if action_name == RESPOND_ACTION_NAME:
                response_text = next((v for v in action_kwargs.values() if str(v).strip()), "")
                if not response_text:
                    PEVLogger.warn("Empty response drafted. Rolling back.")
                    state = checkpoint_state
                    continue
                action_kwargs = {"content": response_text}
                
            action = Action(name=action_name, kwargs=action_kwargs)
            
            PEVLogger.node("Tau-Bench Environment", f"Executing tool: {action.name}")
            env_res = env.step(action)
            
            obs = env_res.observation
            if len(obs) > 500:
                obs = self.engine.distiller.distill_observation(name=action.name, args=action.kwargs, raw_output=obs)
            
            state.last_observation = obs
            state.history.append({"role": "tool", "content": obs})
            obs_lower = obs.lower()
            if obs_lower.startswith("error") or "invalid" in obs_lower:
                status = "ERROR"
            elif obs == "[]" or obs_lower == "false" or obs_lower == "none":
                status = "DATA_MISSING"
            else:
                status = "SUCCESS"
                
            checkpoint_entry = {
                "step": i + 1,
                "action": action.name,
                "status": status,
                "result_summary": obs[:250] + "..." if len(obs) > 250 else obs
            }
            state.manifest.write_ahead_memory.append(checkpoint_entry)
            
            # ASCD/PME Requirement: Record and Progress the Roadmap
            trace_entry = f"f({action.name}({action.kwargs})) -> {obs[:250]}..."
            state.manifest.functional_trace.append(trace_entry)
            
            # Update Roadmap Progress if it was a valid attempt
            current_plan = state.manifest.refined_tactical_plan
            if current_plan in state.manifest.roadmap:
                state.manifest.roadmap_progress[current_plan] = "DONE"
            
            # Update the Safe Checkpoint
            if status in ["SUCCESS", "DATA_MISSING"]:
                checkpoint_state = state
                state.consecutive_errors = 0
            
            PEVLogger.info(f"Observation Length: {len(env_res.observation)} chars")
            
            if env_res.done:
                state.reward = env_res.reward
                PEVLogger.success(f"Task Complete! Reward: {state.reward}")
                break

        # 10. Extract Learned Expertise (Global Reflection)
        # Only save insights if the task failed or was a loop to keep wisdom high-density
        if state.reward < 1.0:
            new_insight = self.global_learner(state)
            self.wisdom_store.save_insight(new_insight)

        return SolveResult(
            reward=state.reward,
            messages=state.history,
            info={}
        )
