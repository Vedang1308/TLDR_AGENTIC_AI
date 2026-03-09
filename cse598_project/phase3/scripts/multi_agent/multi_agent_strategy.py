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
        agent_strategy: str = "multi-agent-react",
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.agent_strategy = agent_strategy
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
        
        # Initialize our PEV State for this specific conversation
        state = PevState(strategy=self.agent_strategy)
        
        # Seed the initial user conversation turn
        state.user_conversation.append({"role": "system", "content": self.wiki})
        state.user_conversation.append({"role": "user", "content": obs})
        
        # Inject tool schemas
        state.tools_info = self.tools_info

        messages_log = state.user_conversation.copy()

        for step_idx in range(max_num_steps):
            print(f"  ▶ Turn {step_idx+1}/{max_num_steps} (Action Cycle)")
            # Run the graph orchestrator
            # This invokes Planner -> Executor -> Monitor -> Validator
            # The graph returns when either the Validator approves an action, 
            # or the Planner determines the task is inherently complete.
            
            try:
                # LangGraph requires state to be passed as a plain dict, not a Pydantic model.
                # We re-hydrate the returned dict back into a PevState object after.
                # Increased recursion limit to 40 (4 nodes per cycle = 10 full retry loops allowed)
                final_state_dict = self.workflow.invoke(state.model_dump(), {"recursion_limit": 40})
            except Exception as e:
                print(f"      ↳ [Orchestrator] Graph recursion limit reached or failed: {e}")
                break
                
            # Re-hydrate returned dict back into our PevState Pydantic model
            state = PevState(**final_state_dict)
                
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
                
                # Push to Context Memory Kernel for the Planner/Validator to see
                state.memory.append({
                    "type": "tool_result",
                    "action_taken": action.name,
                    "arguments_used": action.kwargs,
                    "api_observation": env_response.observation
                })
            else:
                user_resp_msg = {"role": "user", "content": env_response.observation}
                messages_log.append(user_resp_msg)
                state.user_conversation.append(user_resp_msg)
                
            # Clear drafted tool for next loop
            state.drafted_tool_call = None
            
            # CRITICAL: Clear rejection flags! Otherwise, next turn's graph router 
            # will see stale rejection data from 10 turns ago and immediately 
            # jump from Planner -> Rejected -> Planner in an infinite loop!
            state.rejection_feedback = None
            state.rejection_source = None
            
            if env_response.done:
                break
                
        # Inject the specialized node logs so we can extract them later for highlighting
        info["pev_node_logs"] = state.node_logs

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages_log,
            total_cost=total_cost, # Cost tracing is harder across LangGraph, ignoring for now
        )
