import os
import re
import json
import time
from typing import List, Optional, Dict, Any
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from .state import PevState
from .nodes import (
    planner_node, 
    executor_node, 
    syntax_monitor_node, 
    validator_node, 
    translator_node,
    error_reflection_node,
    reformulator_node,
    reflection_strategy_node
)
from .distiller import ContextDistiller

class PEVEngineNative:
    """
    100% Diagram-Compliant Orchestrator. 
    Steps 1-11 implemented with local Gaudi-Native logic.
    """
    def __init__(self, tools_info: List[Dict[str, Any]], wiki: str, log_dir: str = "results/phase3_gaudi_native"):
        self.tools_info = tools_info
        self.wiki = wiki
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Determine Domain from tool metadata if possible, or default to env
        self.domain = os.environ.get("AGENT_DOMAIN", "airline")
        
        # SUMMARIZER (Step 3 in Diagram)
        self.summarizer = ContextDistiller()
        
        # Pre-populate Tools Wiki
        tools_desc = []
        for t in self.tools_info:
            name = t.get('name') or t.get('function', {}).get('name', 'unknown')
            desc = t.get('description') or t.get('function', {}).get('description', 'No description.')
            params = list(t.get('parameters', {}).get('properties', {}).keys()) if 'parameters' in t else list(t.get('function', {}).get('parameters', {}).get('properties', {}).get('parameters', {}).get('properties', {}).keys()) if 'function' in t and 'parameters' in t['function'] else []
            # Safety check for parameter nesting
            if not params and 'function' in t and 'parameters' in t['function']:
                 params = list(t['function']['parameters'].get('properties', {}).keys())

            tools_desc.append(f"- {name}: {desc} (Args: {', '.join(params)})")
        self.tools_wiki = "\n".join(tools_desc)
        
        self.wisdom_file = "results/phase3/persistent_wisdom.json"

    def _load_wisdom(self) -> List[str]:
        if os.path.exists(self.wisdom_file):
            try:
                with open(self.wisdom_file, "r") as f:
                    return json.load(f)
            except: return []
        return []

    def _save_wisdom(self, wisdom: List[str]):
        current = self._load_wisdom()
        updated = list(dict.fromkeys(current + wisdom))
        os.makedirs(os.path.dirname(self.wisdom_file), exist_ok=True)
        with open(self.wisdom_file, "w") as f:
            json.dump(updated, f, indent=2)

    def proactive_seed(self, state: PevState, env: Env, obs: str):
        """Domain-Aware Proactive Context Seeding."""
        # Common: User ID
        user_id_match = re.search(r'\b([a-z]+_[a-z]+_\d{3,6})\b', obs, re.IGNORECASE)
        if user_id_match:
            uid = user_id_match.group(1).lower()
            lookup_tool = next((t for t in self.tools_info if 'user' in (t.get('name') or t.get('function', {}).get('name', '')).lower() and 'detail' in (t.get('name') or t.get('function', {}).get('name', '')).lower()), None)
            if lookup_tool:
                t_name = lookup_tool.get('name') or lookup_tool.get('function', {}).get('name')
                res = env.step(Action(name=t_name, kwargs={"user_id": uid}))
                state.memory.append({"action": "AUTO_PREFETCH", "args": {"user_id": uid}, "observation": str(res.observation)})

        # Domain-Agnostic: Reservation or Order ID
        res_id_match = re.search(r'\b([A-Z\d]{6})\b', obs)
        if res_id_match:
            rid = res_id_match.group(1)
            # Find detail tool that matches either domain
            res_lookup = next((t for t in self.tools_info if (
                'reservation' in (t.get('name') or t.get('function', {}).get('name', '')).lower() or 
                'order' in (t.get('name') or t.get('function', {}).get('name', '')).lower()
            ) and 'detail' in (t.get('name') or t.get('function', {}).get('name', '')).lower()), None)
            
            if res_lookup:
                t_name = res_lookup.get('name') or res_lookup.get('function', {}).get('name')
                arg_name = "reservation_id" if "reservation" in t_name.lower() else "order_id"
                try:
                    res = env.step(Action(name=t_name, kwargs={arg_name: rid}))
                    state.memory.append({"action": "AUTO_PREFETCH", "args": {arg_name: rid}, "observation": str(res.observation)})
                except: pass

    def solve(self, env: Env, task_index: Optional[int] = None, max_steps: int = 30) -> SolveResult:
        env_res = env.reset(task_index=task_index)
        strategy = os.environ.get("AGENT_REASONING_MODE", "fc")
        
        state = PevState(
            tools_info=self.tools_info,
            tools_wiki=self.tools_wiki,
            global_wisdom=self._load_wisdom()
        )
        state.user_conversation.append({"role": "system", "content": self.wiki})
        state.user_conversation.append({"role": "user", "content": env_res.observation})
        
        # Data Hub initialization
        self.proactive_seed(state, env, env_res.observation)
        
        reward = 0.0
        messages_log = state.user_conversation.copy()

        for step in range(max_steps):
            print(f"  [Step {step+1}/{max_steps}] Orchestrating Reasoning Flow...")
            # 1. Distill (Step 3: Strategic Kernel)
            distilled = self.summarizer(state)
            state.strategic_kernel = distilled["summary"]
            state.world_snapshot = distilled["world_snapshot"]

            # [STRATEGY HOOK]: IRMA Reformulation
            if strategy == "irma":
                print(f"  [Step {step+1}] Strategy: IRMA Reformulating Observation...")
                ref_out = reformulator_node(state)
                state.node_logs.extend(ref_out.get("node_logs", []))
                if ref_out.get("reformulated_observation"):
                    state.strategic_kernel = f"### REFORMULATED FOCUS ###\n{ref_out['reformulated_observation']}\n\n{state.strategic_kernel}"

            inner_step = 0
            while inner_step < 10: # Safety cap for internal reasoning
                inner_step += 1
                # 2. Planner (Step 4)
                print(f"  [Step {step+1}.{inner_step}] Node: Planner...")
                p_out = planner_node(state)
                state.node_logs.extend(p_out.get("node_logs", []))
                if p_out.get("task_completed"):
                    state.task_completed = True
                    break
                state.current_plan = p_out.get("current_plan", state.current_plan)

                # 3. Executor (Tactician)
                print(f"  [Step {step+1}.{inner_step}] Node: Executor (Drafting Action)...")
                e_out = executor_node(state)
                state.node_logs.extend(e_out.get("node_logs", []))
                state.drafted_tool_call = e_out.get("drafted_tool_call")

                # 4. Translator (Step 5)
                t_out = translator_node(state)
                state.node_logs.extend(t_out.get("node_logs", []))
                state.drafted_tool_call = t_out.get("drafted_tool_call")

                # 5. Syntax Monitor (Step 6)
                print(f"  [Step {step+1}.{inner_step}] Node: Syntax Monitor...")
                m_out = syntax_monitor_node(state)
                if m_out.get("rejection_feedback"):
                    print(f"    ! REJECTED (Syntax): {m_out['rejection_feedback']}")
                    state.rejection_feedback = m_out["rejection_feedback"]
                    state.rejection_source = m_out["rejection_source"]
                    state.internal_retry_count = m_out["internal_retry_count"]
                    
                    if strategy == "reflection":
                        print(f"    ! Reflecting on Syntax failure...")
                        refl_out = reflection_strategy_node(state)
                        state.error_reflection = refl_out.get("error_reflection")
                        state.node_logs.extend(refl_out.get("node_logs", []))
                    continue 
                
                # 6. Validator (Step 7)
                print(f"  [Step {step+1}.{inner_step}] Node: Validator...")
                v_out = validator_node(state)
                if v_out.get("rejection_feedback"):
                    print(f"    ! REJECTED (Validator): {v_out['rejection_feedback']}")
                    state.rejection_feedback = v_out["rejection_feedback"]
                    state.rejection_source = v_out["rejection_source"]
                    state.internal_retry_count = v_out["internal_retry_count"]
                    
                    if strategy == "reflection":
                        print(f"    ! Reflecting on Validation failure...")
                        refl_out = reflection_strategy_node(state)
                        state.error_reflection = refl_out.get("error_reflection")
                        state.node_logs.extend(refl_out.get("node_logs", []))
                    continue 
                
                break

            if state.task_completed: break
            
            drafted = state.drafted_tool_call
            if not drafted: action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Confused."})
            else: action = Action(name=drafted["name"], kwargs=drafted.get("arguments", {}))
            
            # 7. Dispatch to Env (Step 8)
            print(f"  [Step {step+1}] Dispatching Action: {action.name}")
            res = env.step(action)
            reward = res.reward
            
            # Record Observation
            is_error = any(kw in str(res.observation).lower() for kw in ["error", "invalid", "failed", "not found"])
            state.memory.append({
                "type": "tool_error" if is_error else "tool_result",
                "action_taken": action.name,
                "arguments_used": action.kwargs,
                "api_observation": res.observation
            })
            
            # 8. Learning Node (Step 9/10)
            if is_error:
                state.consecutive_error_count += 1
                if state.consecutive_error_count >= 2:
                    ref_out = error_reflection_node(state)
                    state.error_reflection = ref_out.get("error_reflection")
                    state.failure_log.extend(ref_out.get("failure_log", []))
                    state.consecutive_error_count = 0
            else:
                state.consecutive_error_count = 0
                # Identify User (Airline/Retail Policy Adherence)
                if any(x in action.name.lower() for x in ["get_user", "get_reservation", "get_order"]):
                    if not any(err in str(res.observation).lower() for err in ["not found", "error", "invalid"]):
                        state.user_identified = True
                        print(f"  [POLICY] User successfully identified: {action.kwargs}")

            if action.name == RESPOND_ACTION_NAME:
                state.user_conversation.append({"role": "user", "content": res.observation})
            
            # Anti-Think Loop: If last action was think, inject warning
            if action.name == "think":
                state.rejection_feedback = "You just used the 'think' tool. You MUST now take a physical action or respond to the user to avoid a loop."
                state.rejection_source = "orchestrator"

            if res.done: break
            
        return SolveResult(reward=reward, info=res.info.model_dump(), messages=state.user_conversation, total_cost=0.0)
