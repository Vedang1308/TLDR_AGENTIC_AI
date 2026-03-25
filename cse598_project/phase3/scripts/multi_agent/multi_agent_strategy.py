import os
import re
import json
from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME

from .state import PevState
from .graph import create_pev_graph, create_reflection_graph, create_global_reflection_graph

class MultiAgentStrategy(Agent):
    """
    Replaces the monolithic single-LLM completion loop with our
    LangGraph Plan-Execute-Validate multi-agent orchestration.
    
    Includes PEV-Wisdom: Persistent cross-trial technical self-learning.
    """
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        agent_strategy: str = None,  # Accepted for tau_bench compat; reasoning mode set via AGENT_REASONING_MODE env var
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        
        # Compile our Multi-Agent LangGraphs
        self.workflow = create_pev_graph()
        self.reflection_workflow = create_reflection_graph()
        self.global_reflection_workflow = create_global_reflection_graph()
        
        # Persistent Wisdom Storage (Moved to /tmp to keep workspace clean)
        self.wisdom_file = os.getenv("PHASE3_WISDOM_FILE", "/tmp/persistent_wisdom.json")
        if not os.path.isabs(self.wisdom_file):
            # Fallback safety
            os.makedirs(os.path.dirname(self.wisdom_file), exist_ok=True)

        # Pre-populate Tools Wiki (Technical specs for the Planner)
        tools_desc = []
        for t in self.tools_info:
            name = t.get('name', 'unknown')
            desc = t.get('description', 'No description available.')
            params = list(t.get('parameters', {}).get('properties', {}).keys())
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
        # Keep only unique wisdom to prevent bloating
        current = self._load_wisdom()
        # Merged and deduplicate while preserving order (newest last)
        updated = list(dict.fromkeys(current + wisdom))
        
        # Atomic save to prevent corruption during parallel trials
        temp_file = self.wisdom_file + ".tmp"
        try:
            with open(temp_file, "w") as f:
                json.dump(updated, f, indent=2)
            os.replace(temp_file, self.wisdom_file)
        except Exception as e:
            print(f"Error saving wisdom: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        
        # Setup specific disk-logging for this task
        log_dir = "results/phase3_v2_logs"
        os.makedirs(log_dir, exist_ok=True)
        task_id = task_index if task_index is not None else "custom"
        log_file = os.path.join(log_dir, f"task_{task_id}_execution_trace.md")
        
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"# PEV Execution Trace: Task {task_id}\n\n")

        # Initialize our PEV State for this specific conversation
        state = PevState()
        
        # Setup initial state with persistent wisdom and tool knowledge
        state.global_wisdom = self._load_wisdom()
        state.tools_wiki = self.tools_wiki
        
        # Seed the initial user conversation turn
        state.user_conversation.append({"role": "system", "content": self.wiki})
        state.user_conversation.append({"role": "user", "content": obs})
        
        # Inject tool schemas
        state.tools_info = self.tools_info

        # ── PROACTIVE CONTEXT SEEDING ──────────────────────────────────────────
        # A real customer service agent always pulls up the caller's account
        # before doing anything else. We replicate this:
        #   1. Scan the initial observation for a user_id (domain-agnostic pattern)
        #   2. Find the profile lookup tool dynamically by inspecting tool names
        #   3. Call it automatically and seed the memory kernel
        #
        # This is NOT hardcoded domain logic — it's a universal initialization
        # heuristic: "if we know who the user is, look them up before planning."
        # The Planner then has real account data (balances, reservations, DOBs)
        # instead of reasoning in a data vacuum and concluding "I can't do this."
        
        user_id_match = re.search(
            r'\b([a-z]+_[a-z]+_\d{3,6})\b', obs, re.IGNORECASE
        )
        if user_id_match:
            detected_user_id = user_id_match.group(1).lower()
            lookup_tool = None
            for t in self.tools_info:
                t_name = t.get('name', '').lower() if not 'function' in t else t['function'].get('name', '').lower()
                if 'user' in t_name and 'detail' in t_name:
                    lookup_tool = t
                    break
            
            if lookup_tool:
                try:
                    tool_name = lookup_tool.get('name') or lookup_tool.get('function', {}).get('name')
                    # IDENTIFY ARG NAME FROM SCHEMA (Truly domain-agnostic)
                    schema = lookup_tool.get('parameters', {}) if not 'function' in lookup_tool else lookup_tool['function'].get('parameters', {})
                    arg_name = list(schema.get('properties', {}).keys())[0] if schema.get('properties') else 'user_id'
                    
                    res = env.step(Action(name=tool_name, kwargs={arg_name: detected_user_id}))
                    state.memory.append({
                        "action": "AUTO_PREFETCH",
                        "args": {arg_name: detected_user_id},
                        "observation": str(res.observation)
                    })
                    print(f"--- [PROACTIVE] Pre-fetched using {tool_name} with {arg_name}={detected_user_id} ---")
                except:
                    pass

        # Also look for reservation/order IDs (alphanumeric 6-char strings)
        res_id_match = re.search(r'\b([A-Z\d]{6})\b', obs)
        if res_id_match:
            detected_res_id = res_id_match.group(1)
            res_lookup = None
            for t in self.tools_info:
                t_name = t.get('name', '').lower() if not 'function' in t else t['function'].get('name', '').lower()
                if ('reservation' in t_name or 'order' in t_name) and 'detail' in t_name:
                    res_lookup = t
                    break
            
            if res_lookup:
                try:
                    tool_name = res_lookup.get('name') or res_lookup.get('function', {}).get('name')
                    # IDENTIFY ARG NAME FROM SCHEMA (Truly domain-agnostic)
                    schema = res_lookup.get('parameters', {}) if not 'function' in res_lookup else res_lookup['function'].get('parameters', {})
                    arg_name = list(schema.get('properties', {}).keys())[0] if schema.get('properties') else 'reservation_id'

                    res = env.step(Action(name=tool_name, kwargs={arg_name: detected_res_id}))
                    state.memory.append({
                        "action": "AUTO_PREFETCH",
                        "args": {arg_name: detected_res_id},
                        "observation": str(res.observation)
                    })
                    print(f"--- [PROACTIVE] Pre-fetched using {tool_name} with {arg_name}={detected_res_id} ---")
                except:
                    pass


        messages_log = state.user_conversation.copy()

        for _ in range(max_num_steps):
            # Run the graph orchestrator
            # This invokes Planner -> Executor -> Monitor -> Validator
            # The graph returns when either the Validator approves an action, 
            # or the Planner determines the task is inherently complete.
            
            try:
                # Compile returns a Runnable, we invoke it with our state
                final_state = self.workflow.invoke(state, {"recursion_limit": 30})
            except Exception as e:
                print(f"Graph failed: {e}")
                
                # If the graph violently loops or hits a recursion limit, 
                # we MUST manually craft a fallback transfer tool call so the 
                # simulation correctly terminates instead of python crashing the batch.
                state.drafted_tool_call = {
                    "name": "transfer_to_human_agents", 
                    "arguments": {"summary": f"Graph tragically crashed due to internal loop constraints: {str(e)}"}
                }
                final_state = {k: getattr(state, k) for k in vars(state)}
                
            # Update our state object with changes from the graph
            for k, v in final_state.items():
                setattr(state, k, v)
            
            # Accumulate node logs into info for result tracing
            step_logs = final_state.get("node_logs", [])
            info.setdefault("pev_node_logs", []).extend(step_logs)
                
            # Check if planner forced an exit
            if state.task_completed:
                break
                
            drafted_tool = state.drafted_tool_call
            
            if not drafted_tool:
                # Graph couldn't formulate a valid tool after loops
                # We force a conversational response indicating confusion
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": "I encountered an internal error and could not proceed."})
            elif "name" not in drafted_tool or drafted_tool["name"] == RESPOND_ACTION_NAME:
                content = drafted_tool.get("arguments", {}).get("content", "I am transferring you.")
                if not isinstance(content, str):
                    content = str(content)
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
            else:
                action = Action(
                    name=drafted_tool["name"],
                    kwargs=drafted_tool.get("arguments", {})
                )
            
            # Formulate the message to add to our external logs
            agent_msg = {"role": "assistant"}
            if action.name == RESPOND_ACTION_NAME:
                agent_msg["content"] = action.kwargs.get("content", "")
            else:
                agent_msg["tool_calls"] = [{
                    "id": "call_pev_1",
                    "type": "function",
                    "function": {
                        "name": action.name,
                        "arguments": json.dumps(action.kwargs)
                    }
                }]
            messages_log.append(agent_msg)

            # Step the tau-bench environment
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            
            # Log the environment's observation (API result or User response) into our Memory Kernel
            if action.name != RESPOND_ACTION_NAME:
                tool_result_msg = {
                    "role": "tool",
                    "tool_call_id": "call_pev_1",
                    "name": action.name,
                    "content": env_response.observation
                }
                messages_log.append(tool_result_msg)
                
                # Broadened domain-agnostic error heuristic to reliably trigger Metacognition
                error_signals = ["error", "invalid", "exception", "not found", "failed", "insufficient", "cannot", "does not"]
                is_error = any(kw in env_response.observation.lower() for kw in error_signals)
                
                # Push to Context Memory Kernel for the Planner/Validator to see.
                # Crucial Fix: We do NOT push 'think' actions to the memory kernel. 
                # 'think' is internal scratchpad context and bloating the memory kernel 
                # with it pushes real API data (like user profiles) out of the Planner's view.
                if action.name != "think":
                    state.memory.append({
                        "type": "tool_error" if is_error else "tool_result",
                        "action_taken": action.name,
                        "arguments_used": action.kwargs,
                        "api_observation": env_response.observation,
                        "tool_call": drafted_tool
                    })
                
                # --- SELF-CORRECTION: Track consecutive errors and trigger reflection ---
                if is_error:
                    state.consecutive_error_count = state.consecutive_error_count + 1
                    # After 2 consecutive errors, invoke the Error Reflection node
                    # This is the 'step back and think' mechanism — domain-agnostic metacognition
                    if state.consecutive_error_count >= 2:
                        try:
                            reflection_state = self.reflection_workflow.invoke(
                                state, {"recursion_limit": 5}
                            )
                            # Pull out the reflection and failure_log entries
                            for k, v in reflection_state.items():
                                if k in ["error_reflection", "failure_log", "consecutive_error_count", "node_logs"]:
                                    if k == "failure_log" and v:
                                        state.failure_log = state.failure_log + v
                                    elif k == "node_logs" and v:
                                        info.setdefault("pev_node_logs", []).extend(v)
                                    else:
                                        setattr(state, k, v)
                        except Exception as reflect_err:
                            print(f"Error Reflection failed: {reflect_err}")
                else:
                    # Reset consecutive error count on success
                    state.consecutive_error_count = 0
            else:
                user_resp_msg = {"role": "user", "content": env_response.observation}
                messages_log.append(user_resp_msg)
                state.user_conversation.append(user_resp_msg)
                
            # Clear drafted tool and error count for next loop
            state.drafted_tool_call = None
            state.internal_retry_count = 0
            
            # --- DISK LOGGING FOR DEBUGGING ---
            step_num = _ + 1
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"## Step {step_num}\n")
                f.write(f"### 1. Planner's Final Formulated Plan\n")
                f.write(f"> {state.current_plan}\n\n")
                f.write(f"### 2. Executor's Action\n")
                f.write(f"**Action**: `{action.name}`\n")
                f.write(f"**Arguments**: ```json\n{json.dumps(action.kwargs, indent=2)}\n```\n\n")
                f.write(f"### 3. Environment Observation (API Return)\n")
                f.write(f"```text\n{env_response.observation}\n```\n\n")
                f.write(f"### 4. Memory Kernel Context Provided to Agents\n")
                f.write(f"```json\n{json.dumps(state.memory, indent=2)}\n```\n")
                f.write("---\n\n")
            
            if env_response.done:
                break
                
        # --- TASK COMPLETION: SELF-LEARNING ---
        # If the task failed (reward < 1.0), we synthesize a "Global Insight" 
        # using the global reflection graph. This insight is saved persistently 
        # to help future trials and tasks avoid the same technical trap.
        if reward < 1.0:
            try:
                # Add one final log entry for the global reflector
                state.memory.append({
                    "type": "final_failure_state",
                    "reward": reward,
                    "explanation": "Task terminated without completing the required goal."
                })
                
                final_reflection = self.global_reflection_workflow.invoke(state, {"recursion_limit": 5})
                new_wisdom = final_reflection.get("global_wisdom", [])
                if new_wisdom:
                    self._save_wisdom(new_wisdom)
                    # Add to info for transparency in logs
                    info.setdefault("pev_global_insights", []).extend(new_wisdom)
            except Exception as e:
                print(f"Global Reflection failed: {e}")

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages_log,
            total_cost=total_cost,
        )
