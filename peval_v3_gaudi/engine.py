import os
import json
from typing import List, Dict, Any, Optional
from peval_v3_gaudi.state import PevState
from peval_v3_gaudi.nodes import (
    planner_node, executor_node, validator_node, 
    error_reflection_node, global_reflector_node,
    proactive_prefetch, strategic_auditor_node
)
from peval_v4_lite.src.core.model_client import ModelClient
from peval_v4_lite.src.core.logger import PEVLogger
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME

class PEVEngine:
    """
    Non-LangGraph Orchestrator for Phase 3 Gaudi-Lite.
    Implements the successful 'Plan-Execute-Verify' loop in pure Python.
    """
    def __init__(self, tools_info: List[Dict[str, Any]], wiki: str, log_dir: str = "results/phase3_gaudi"):
        self.tools_info = tools_info
        self.wiki = wiki
        self.log_dir = log_dir
        self.wisdom_file = os.path.join(log_dir, "persistent_wisdom_gaudi.json")
        os.makedirs(log_dir, exist_ok=True)
        
        # Pre-build Tools Wiki
        tools_desc = []
        for t in self.tools_info:
            name = t.get('name') or t.get('function', {}).get('name', 'unknown')
            desc = t.get('description') or t.get('function', {}).get('description', 'No description.')
            params = list(t.get('parameters', {}).get('properties', {}).keys()) if 'parameters' in t else list(t.get('function', {}).get('parameters', {}).get('properties', {}).keys())
            tools_desc.append(f"- {name}: {desc} (Args: {', '.join(params)})")
        self.tools_wiki = "\n".join(tools_desc)

    def _load_wisdom(self) -> List[str]:
        if os.path.exists(self.wisdom_file):
            try:
                with open(self.wisdom_file, "r") as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_wisdom(self, wisdom: List[str]):
        current = self._load_wisdom()
        updated = list(dict.fromkeys(current + wisdom))
        with open(self.wisdom_file, "w") as f:
            json.dump(updated, f, indent=2)

    def solve(self, env, task_index: int, max_steps: int = 30) -> SolveResult:
        PEVLogger.info(f"--- [START] Solving Task {task_index} with Phase 3 Gaudi-Lite ---")
        
        env_res = env.reset(task_index=task_index)
        state = PevState(
            user_conversation=[
                {"role": "system", "content": self.wiki},
                {"role": "user", "content": env_res.observation}
            ],
            tools_info=self.tools_info,
            tools_wiki=self.tools_wiki,
            global_wisdom=self._load_wisdom(),
            wisdom_file=self.wisdom_file
        )

        # 1. Proactive Pre-fetch (Phase 3 Heuristic)
        proactive_prefetch(env, state)

        reward = 0.0
        info = env_res.info.model_dump()
        messages_log = state.user_conversation.copy()

        for step in range(max_steps):
            state.rejection_feedback = None # Clear for fresh step
            PEVLogger.info(f"=== STEP {step+1} ===")
            
            # --- PHASE 4: STRATEGIC AUDIT ---
            audit_res = strategic_auditor_node(state)
            state.strategic_objective = audit_res.get("strategic_objective", "")
            state.node_logs.extend(audit_res.get("node_logs", []))

            # INNER LOOP: Planner -> Executor -> Validator (Retry up to 3 times if rejected)
            inner_retries = 0
            while inner_retries < 3:
                # Node 1: Planner
                planner_res = planner_node(state)
                state.current_plan = planner_res.get("current_plan", state.current_plan)
                state.task_completed = planner_res.get("task_completed", False)
                state.node_logs.extend(planner_res.get("node_logs", []))

                if state.task_completed:
                    break

                # Node 2: Executor
                exec_res = executor_node(state)
                state.drafted_tool_call = exec_res.get("drafted_tool_call")
                state.node_logs.extend(exec_res.get("node_logs", []))

                # Node 3: Validator
                val_res = validator_node(state)
                state.rejection_feedback = val_res.get("rejection_feedback")
                state.rejection_source = val_res.get("rejection_source")
                state.node_logs.extend(val_res.get("node_logs", []))

                if not state.rejection_feedback:
                    break # Approved!
                
                PEVLogger.warn(f"Rejection: {state.rejection_feedback}")
                inner_retries += 1
            
            # --- ELITE: 3-STRIKE FALLBACK ---
            if state.rejection_feedback and inner_retries >= 3:
                PEVLogger.error("CRITICAL: Inner loop failed to resolve rejection. Falling back to Respond and FLUSHING PLAN.")
                state.drafted_tool_call = {"name": "respond", "arguments": {"content": f"I'm having difficulty finalizing a valid next step. Let me step back and reassess the task."}}
                state.rejection_feedback = None # Clear after fallback
                state.current_plan = "" # ELITE: Reset the plan to break the reasoning loop
            
            if state.task_completed:
                PEVLogger.success("Planner marked task as completed.")
                break

            # Execute the approved/final drafted tool
            drafted = state.drafted_tool_call
            if not drafted:
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": "I apologize, I am having trouble formulating my next step."})
            elif drafted.get("name") == "respond" or "content" in drafted.get("arguments", {}):
                content = drafted.get("arguments", {}).get("content") or drafted.get("content") or "How can I help you?"
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
            else:
                action = Action(name=drafted["name"], kwargs=drafted.get("arguments", {}))

            # --- ELITE LOGGING: ACTION ---
            readable_args = json.dumps(action.kwargs, indent=None)
            PEVLogger.success(f"Action: {action.name} | Args: {readable_args}")

            # Log to messages
            if action.name == RESPOND_ACTION_NAME:
                messages_log.append({"role": "assistant", "content": action.kwargs.get("content", "")})
            else:
                messages_log.append({
                    "role": "assistant", 
                    "tool_calls": [{"id": f"call_{step}", "type": "function", "function": {"name": action.name, "arguments": json.dumps(action.kwargs)}}]
                })

            # Env Step
            env_res = env.step(action)
            
            # --- ELITE: PLAN FLUSH ON ERROR ---
            # If the environment returns an error, we flush the plan so the Planner MUST reconsider its strategy.
            if "Error" in str(env_res.observation) or "not found" in str(env_res.observation).lower() or "not match" in str(env_res.observation).lower():
                PEVLogger.warn("Environment Error detected. FLUSHING PLAN to force recalibration.")
                state.current_plan = ""
            
            # --- ELITE LOGGING: OBSERVATION ---
            obs_clean = str(env_res.observation)[:500] + "..." if len(str(env_res.observation)) > 500 else str(env_res.observation)
            PEVLogger.info(f"Observation: {obs_clean}")

            reward = env_res.reward
            info = {**info, **env_res.info.model_dump()}

            # Memory Kernel Update
            if action.name != RESPOND_ACTION_NAME:
                # --- ELITE PREFIXING ---
                obs = "API output: " + str(env_res.observation)
                messages_log.append({"role": "tool", "tool_call_id": f"call_{step}", "name": action.name, "content": obs})
                
                is_error = any(kw in obs.lower() for kw in ["error", "invalid", "fail", "not found"])
                state.memory.append({
                    "type": "tool_error" if is_error else "tool_result",
                    "action_taken": action.name,
                    "arguments_used": action.kwargs,
                    "api_observation": obs
                })

                if is_error:
                    state.consecutive_error_count += 1
                    if state.consecutive_error_count >= 2:
                        reflect_res = error_reflection_node(state)
                        state.error_reflection = reflect_res.get("error_reflection")
                        state.failure_log = reflect_res.get("failure_log", state.failure_log)
                else:
                    state.consecutive_error_count = 0
            else:
                messages_log.append({"role": "user", "content": env_res.observation})
                state.user_conversation.append({"role": "user", "content": env_res.observation})

            # --- ELITE: CONTEXT DISTILLATION (Every 10 steps) ---
            if (step + 1) % 10 == 0 and len(state.memory) > 5:
                PEVLogger.warn("Context Distillation: Compressing memory into Situational Report...")
                summarizer = ModelClient(mode="summarizer")
                history_text = "\n".join([f"Step {i}: {m.get('action_taken')} -> {str(m.get('api_observation'))[:200]}" for i, m in enumerate(state.memory)])
                sys_summary = f"### SITUATIONAL INVENTORY ###\nCompare the history against the original user request and output a clean summary of what is KNOWN and what is MISSING. Identify exactly what data has been retrieved (e.g., reservation_ids, prices, passenger_details). This report is the source of truth for the NEXT STEP selection.\n\nCurrent History:\n{history_text}"
                report = summarizer.chat([{"role": "system", "content": sys_summary}])
                
                # Prepend the report to the tool wiki or as a special memory entry
                state.tools_wiki = f"### SITUATIONAL REPORT (Steps 1-{step+1}):\n{report}\n\n" + self.tools_wiki
                PEVLogger.info(f"SitRep Generated: {report[:100]}...")

            if env_res.done:
                break

        # Post-task Reflection (Learning)
        if reward < 1.0:
            reflect_res = global_reflector_node(state)
            new_wisdom = reflect_res.get("global_wisdom", [])
            if new_wisdom:
                self._save_wisdom(new_wisdom)
                PEVLogger.info(f"Learned Global Wisdom: {new_wisdom[-1]}")

        return SolveResult(reward=reward, info=info, messages=messages_log)
