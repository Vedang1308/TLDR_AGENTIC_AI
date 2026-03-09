import os
import json
from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME

from .state import PevState
from .graph import create_pev_graph

class MultiAgentStrategy(Agent):
    """
    Replaces the monolithic single-LLM completion loop with our
    LangGraph Plan-Execute-Validate multi-agent orchestration.
    """
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        
        # Compile our Multi-Agent LangGraph
        self.workflow = create_pev_graph()

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
        
        # Seed the initial user conversation turn
        state.user_conversation.append({"role": "system", "content": self.wiki})
        state.user_conversation.append({"role": "user", "content": obs})
        
        # Inject tool schemas
        state.tools_info = self.tools_info

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
                
                is_error = any(kw in env_response.observation.lower() for kw in ["error", "invalid", "exception"])
                # Push to Context Memory Kernel for the Planner/Validator to see
                state.memory.append({
                    "type": "tool_error" if is_error else "tool_result",
                    "action_taken": action.name,
                    "arguments_used": action.kwargs,
                    "api_observation": env_response.observation,
                    "tool_call": drafted_tool
                })
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
                
        # We deliberately DO NOT inject node_logs into the 'info' array here, 
        # so that the main tau-bench wrapper's terminal output remains clean entirely.
        # Logs are perfectly safely written to the Markdown files on disk!

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages_log,
            total_cost=total_cost, # Cost tracing is harder across LangGraph, ignoring for now
        )
