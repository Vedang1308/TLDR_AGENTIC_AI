import os
import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .state import PevState

# The port map determines which model to call locally:
# 8000: Agent (4B, 8B, 14B, 32B)
# 8001: User (fixed to 32B)
# From run_phase1_experiments.py we know we passed `--model-provider openai` 
# with base_url injected, but here we can instantiate directly.

def get_llm(context="agent"):
    # Read custom env vars passed down from phase3 run script
    # We'll set these dynamically when orchestrating.
    api_base = os.environ.get("AGENT_API_BASE", "http://localhost:8000/v1")
    model_name = os.environ.get("AGENT_MODEL_NAME", "Qwen/Qwen3-4B")
    
    return ChatOpenAI(
        base_url=api_base,
        api_key="EMPTY",
        model=model_name,
        temperature=0.0,
        max_tokens=512,
        stop=["Observation:", "OBSERVATION:"]
    )

def planner_node(state: PevState) -> Dict:
    """
    Supervisor Node: Analyzes conversation, checks memory, sets next objective.
    Does NOT output JSON tool calls. Outputs natural language plan.
    """
    llm = get_llm()
    
    # 1. Build context
    sys_prompt = """You are the HIERARCHICAL PLANNER for a tool-using agent in a customer service setting.
Your job is to read the user conversation and the memory kernel, then output a strict, 1-2 sentence PLAN for the Executor.
DO NOT attempt to formulate JSON tool calls. 
DO NOT converse with the user unless the task requires asking for clarification.
CRITICAL TAU-BENCH RULE: Do NOT transfer to a human agent unless you have exhausted all other options or the user explicitly demands it. Transferring results in immediate failure. Try your best to solve the issue using the API tools first.
If the task is complete, output exactly [TASK COMPLETED].
If an action just failed (see Rejection Feedback), adjust the plan to bypass the error.

MEMORY KERNEL:
{memory_str}

REJECTION FEEDBACK:
{feedback}
"""

    mem_str = json.dumps(state.memory, indent=2) if state.memory else "No prior history."
    feed_str = f"Source: {state.rejection_source} | Message: {state.rejection_feedback}" if state.rejection_feedback else "None."
    
    sys_msg = SystemMessage(content=sys_prompt.format(memory_str=mem_str, feedback=feed_str))
    
    # Format user convo
    # Always include the system wiki (turn 0) which contains business logic!
    user_msgs = []
    if state.user_conversation:
        user_msgs.append(SystemMessage(content=state.user_conversation[0]['content']))
        
    # include only the most recent conversation to prevent context explosion
    for turn in state.user_conversation[-3:]: 
        if turn['role'] != 'system': # don't duplicate the wiki if it's in the last 3
            user_msgs.append(HumanMessage(content=f"{turn['role']}: {turn['content']}"))
        
    response = llm.invoke([sys_msg] + user_msgs)
    plan = response.content.strip()
    
    if "[TASK COMPLETED]" in plan:
        return {"task_completed": True, "node_logs": [{"node": "planner", "plan": plan}]}
        
    return {
        "current_plan": plan, 
        "rejection_feedback": None, # Clear it once incorporated into plan
        "rejection_source": None,
        "node_logs": [{"node": "planner", "plan": plan}]
    }

def executor_node(state: PevState) -> Dict:
    """
    Actor Node: Receives the specific plan and drafts the exact JSON tool call.
    """
    llm = get_llm()
    
    sys_prompt = f"""You are the EXECUTOR. 
Your ONLY job is to output a raw JSON dictionary representing a function call based on the PLAN provided.
Do NOT write explanations or think markers. Just output the JSON.
Available tools:
{json.dumps(state.tools_info, indent=2)}
Additionally, you have access to `transfer_to_human_agents` with arguments {{"summary": "<str>"}} or {{"content": "<str>"}}.

PLAN TO EXECUTE:
{state.current_plan}

MEMORY CONTEXT:
{json.dumps([m for m in state.memory if m.get('type') == 'tool_call'], indent=2)}
"""

    resp = llm.invoke([SystemMessage(content=sys_prompt)])
    content = resp.content.strip()
    
    # Very naive extraction, we will refine this
    import re
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    
    drafted_tool = None
    if json_match:
        try:
            drafted_tool = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
            
    return {"drafted_tool_call": drafted_tool, "node_logs": [{"node": "executor", "raw_output": content}]}

def syntax_monitor_node(state: PevState) -> Dict:
    """
    Code Monitor: Deterministically checks syntax and repetition.
    """
    tool_draft = state.drafted_tool_call
    
    if not tool_draft:
        return {"rejection_feedback": "Executor failed to output valid JSON. Try again.", "rejection_source": "syntax_monitor"}
        
    if "name" not in tool_draft or "arguments" not in tool_draft:
        return {"rejection_feedback": "JSON missing 'name' or 'arguments' keys.", "rejection_source": "syntax_monitor"}
        
    # Check for repetitive loop (did we try this EXACT tool call and fail recently?)
    recent_failures = [m for m in state.memory[-5:] if m.get('type') == 'tool_error']
    for rf in recent_failures:
        if rf.get('tool_call') == tool_draft:
            return {"rejection_feedback": "You already tried this exact action and it failed. Formulate a NEW strategy.", "rejection_source": "syntax_monitor"}

    # Passed
    return {"node_logs": [{"node": "syntax_monitor", "status": "passed"}]}

def validator_node(state: PevState) -> Dict:
    """
    Critic Node: Evaluates drafted tool against business logic.
    """
    llm = get_llm()
    
    sys_prompt = f"""You are the VALIDATOR.
Review the following drafted tool call for logic and preconditions.
If it violates policy (e.g. refunding to wrong payment type, or acting without searching first), output "REJECT: <reason>".
If the tool is `transfer_to_human_agents` but there are still viable APIs to try to solve the user's issue, output "REJECT: Attempt to solve the issue yourself before transferring."
If it is safe and logically sound, output "APPROVE".

DRAFTED TOOL:
{json.dumps(state.drafted_tool_call, indent=2)}

PRIOR MEMORY (to check preconditions):
{json.dumps(state.memory, indent=2)}
"""

    resp = llm.invoke([SystemMessage(content=sys_prompt)])
    decision = resp.content.strip()
    
    if decision.startswith("REJECT"):
        return {"rejection_feedback": decision, "rejection_source": "validator"}
        
    # If approved, we don't execute it here. We just mark it approved for the orchestrator to pass to Tau-Bench.
    return {"node_logs": [{"node": "validator", "status": "approved"}]}
