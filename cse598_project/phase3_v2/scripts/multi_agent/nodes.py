import os
import re
import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .state import PevState

def get_llm(context="agent"):
    api_base = os.environ.get("AGENT_API_BASE", "http://localhost:8000/v1")
    model_name = os.environ.get("AGENT_MODEL_NAME", "Qwen/Qwen3-4B")
    
    return ChatOpenAI(
        base_url=api_base,
        api_key="EMPTY",
        model=model_name,
        temperature=0.0,
        max_tokens=1500,
        stop=["Observation:", "OBSERVATION:"]
    )

def format_memory(memory_list: List[Dict]) -> str:
    """Converts the raw JSON memory array into a clean, human-readable structural markdown trace for the LLM."""
    if not memory_list:
        return "No prior history."
    out = []
    for i, m in enumerate(memory_list):
        if m.get('type') == 'tool_result':
            out.append(f"--- Step {i+1} ---")
            out.append(f"Action: {m.get('action_taken')}")
            out.append(f"Arguments: {json.dumps(m.get('arguments_used', {}))}")
            out.append(f"Result Observation: {str(m.get('api_observation', 'None'))}") # Let the LLM see the whole JSON returned
        elif m.get('type') == 'tool_error':
            out.append(f"--- Step {i+1} [FAILED] ---")
            out.append(f"Attempted Action: {m.get('action_taken')}")
            out.append(f"Error: {m.get('api_observation', 'None')}")
    return "\n".join(out) if out else "No parseable actions."

def invoke_with_paradigm(llm, sys_prompt: str, user_msgs: List, tools: List[Dict], reasoning_mode: str, role_name: str):
    """
    Universal wrapper to force ANY agent (Planner/Executor/Validator) 
    to output using Act, ReAct, or strict Function Calling (FC) with Claude-optimized prompts.
    """
    example_tools = '''
[
  {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state"}
            },
            "required": ["location"]
        }
    }
  }
]
'''
    
    if reasoning_mode == "react":
        instruction = f"""
<tool_instructions>
You are an expert agent. You must use the provided tools to assist the user.
At each step, your generation MUST follow this exact sequence:

1. Think step-by-step about what to do next inside a Thought block.
2. Output a valid JSON Action block containing the tool execution.

**Format required:**
Thought:
<A single line of powerful reasoning to analyze your current task, state the chosen tool, and map its required arguments.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

**Example Usage for a Weather query:**
Thought:
Since the user asked for the weather in San Francisco, I need to use the get_current_weather tool with the location parameter set.
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA"}}}}

CRITICAL: The Action must be perfectly valid JSON with no trailing commas.
</tool_instructions>
"""
        sys_prompt_final = sys_prompt + "\n" + instruction + "\n<available_tools>\n" + json.dumps(tools, indent=2) + "\n</available_tools>\n"
        resp = llm.invoke([SystemMessage(content=sys_prompt_final)] + user_msgs)
        content = resp.content.strip()
        
        # Robust multi-block extraction
        try:
            # 1. Try to strictly parse the Action block ignoring Thought brackets
            action_split = content.split("Action:")[-1].strip()
            start = action_split.find('{')
            end = action_split.rfind('}')
            if start != -1 and end != -1:
                return json.loads(action_split[start:end+1]), content
        except json.JSONDecodeError:
            pass
            
        # 2. Fallback extreme extraction (regex)
        match = re.search(r'\{[^{}]*\"name\"[^{}]*\}', content)
        if match:
            try:
                return json.loads(match.group(0)), content
            except json.JSONDecodeError:
                pass
                
        return None, content

    elif reasoning_mode == "act":
        instruction = f"""
<tool_instructions>
You are an expert agent. You must use the provided tools to assist the user.
You MUST output your decision as a raw Action block. Do NOT write any conversational text or thinking out loud.

**Format required:**
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

**Example Usage for a Weather query:**
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA"}}}}

CRITICAL: Your entire generation must instantly be the Action block. It must be valid JSON.
</tool_instructions>
"""
        sys_prompt_final = sys_prompt + "\n" + instruction + "\n<available_tools>\n" + json.dumps(tools, indent=2) + "\n</available_tools>\n"
        resp = llm.invoke([SystemMessage(content=sys_prompt_final)] + user_msgs)
        content = resp.content.strip()
        
        try:
            action_split = content.split("Action:")[-1].strip()
            start = action_split.find('{')
            end = action_split.rfind('}')
            if start != -1 and end != -1:
                return json.loads(action_split[start:end+1]), content
        except json.JSONDecodeError:
            pass
            
        match = re.search(r'\{[^{}]*\"name\"[^{}]*\}', content)
        if match:
            try:
                return json.loads(match.group(0)), content
            except json.JSONDecodeError:
                pass

        return None, content

    else: # "fc" Native Function Calling
        instruction = """
<tool_instructions>
You are operating in Native Function Calling mode. 
Your ONLY job is to output a raw JSON dictionary representing a function call based on your task. 
Do NOT write any text, conversational filler, or <think> tags. Just output the mathematical JSON payload.
</tool_instructions>
"""
        sys_prompt_final = sys_prompt + "\n" + instruction
        llm_with_tools = llm.bind_tools(tools)
        resp = llm_with_tools.invoke([SystemMessage(content=sys_prompt_final)] + user_msgs)
        
        if hasattr(resp, 'tool_calls') and resp.tool_calls:
            tc = resp.tool_calls[0]
            return {"name": tc["name"], "arguments": tc["args"]}, "Tool Call: " + tc["name"]
        else:
            # Fallback manually parsing
            content = resp.content.strip()
            try:
                action_split = content.split("Action:")[-1].strip() if "Action:" in content else content
                start = action_split.find('{')
                end = action_split.rfind('}')
                if start != -1 and end != -1:
                    parsed = json.loads(action_split[start:end+1])
                    if "name" in parsed:
                        return parsed, content
            except json.JSONDecodeError:
                pass
                
            match = re.search(r'\{[^{}]*\"name\"[^{}]*\}', content)
            if match:
                try:
                    return json.loads(match.group(0)), content
                except json.JSONDecodeError:
                    pass
                    
            return None, content

def planner_node(state: PevState) -> Dict:
    llm = get_llm()
    reasoning_mode = os.environ.get("AGENT_REASONING_MODE", "fc")
    
    sys_prompt = """You are the HIERARCHICAL PLANNER for a customer service agent.
Your job is to read the user conversation and memory kernel, and use the 'submit_plan' tool to set the objective.
CRITICAL TAU-BENCH RULE 1: Do NOT transfer to a human agent unless you have exhausted all other options.
CRITICAL TAU-BENCH RULE 2: If the user forgets a required parameter (like a `reservation_id` or `user_id`), your plan MUST be to use the `respond` tool to ask the user for it. Do NOT panic and transfer.
CRITICAL TAU-BENCH RULE 3: You must handle missing information conversationally.

MEMORY KERNEL:
{memory_str}

REJECTION FEEDBACK:
{feedback}
"""
    mem_str = format_memory(state.memory)
    feed_str = f"Source: {state.rejection_source} | Message: {state.rejection_feedback}" if state.rejection_feedback else "None."
    sys_prompt = sys_prompt.format(memory_str=mem_str, feedback=feed_str)

    user_msgs = []
    if state.user_conversation:
        user_msgs.append(SystemMessage(content=state.user_conversation[0]['content']))
    for turn in state.user_conversation[-3:]: 
        if turn['role'] != 'system':
            user_msgs.append(HumanMessage(content=f"{turn['role']}: {turn['content']}"))

    tools = [{
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": "Submit a 1-2 sentence plan for the executor",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {"type": "string", "description": "The plan to execute next."},
                    "task_completed": {"type": "boolean", "description": "True ONLY if the user's issue is fully resolved."}
                },
                "required": ["plan", "task_completed"]
            }
        }
    }]
    
    parsed_json, raw_log = invoke_with_paradigm(llm, sys_prompt, user_msgs, tools, reasoning_mode, "Planner")
    
    if parsed_json and parsed_json.get("name") == "submit_plan":
        args = parsed_json.get("arguments", {})
        plan = args.get("plan", "Follow the rules.")
        if args.get("task_completed", False):
            return {"task_completed": True, "node_logs": [{"node": "planner", "plan": plan, "log": raw_log}]}
        return {
            "current_plan": plan, 
            "rejection_feedback": None,
            "rejection_source": None,
            "node_logs": [{"node": "planner", "plan": plan, "log": raw_log}]
        }
    else:
        # Fallback if planning fails
        return {
            "current_plan": "Proceed with default action or ask for clarification.",
            "rejection_feedback": None,
            "rejection_source": None,
            "node_logs": [{"node": "planner", "error": "Failed to output plan", "log": raw_log}]
        }

def executor_node(state: PevState) -> Dict:
    llm = get_llm()
    reasoning_mode = os.environ.get("AGENT_REASONING_MODE", "fc")
    
    sys_prompt = f"""You are the EXECUTOR.
Your ONLY job is to select the exact tool call based on the PLAN provided.

PLAN TO EXECUTE:
{state.current_plan}

MEMORY CONTEXT (Recent past actions):
{format_memory(state.memory)}
"""
    if state.rejection_feedback and state.rejection_source == "syntax_monitor":
        sys_prompt += f"\n\n[CRITICAL]: YOUR PREVIOUS DRAFT WAS REJECTED. Fix this error:\n{state.rejection_feedback}\n"


    tools = state.tools_info.copy()
    tools.append({
        "type": "function",
        "function": {
            "name": "transfer_to_human_agents",
            "description": "Transfer to human agent.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}}
            }
        }
    })

    parsed_json, raw_log = invoke_with_paradigm(llm, sys_prompt, [], tools, reasoning_mode, "Executor")
    return {"drafted_tool_call": parsed_json, "node_logs": [{"node": "executor", "raw_output": raw_log}]}

def syntax_monitor_node(state: PevState) -> Dict:
    tool_draft = state.drafted_tool_call
    current_retries = state.internal_retry_count + 1
    
    if current_retries >= 5:
        fallback = {"name": "respond", "arguments": {"content": "I encountered an internal logic error and could not proceed. Can you rephrase or try another approach?"}}
        return {"drafted_tool_call": fallback, "internal_retry_count": 0, "rejection_feedback": None, "rejection_source": None}
        
    if not tool_draft:
        return {"rejection_feedback": "Executor failed to output valid JSON. Try again.", "rejection_source": "syntax_monitor", "internal_retry_count": current_retries}
    if "name" not in tool_draft or "arguments" not in tool_draft:
        return {"rejection_feedback": "JSON missing 'name' or 'arguments' keys.", "rejection_source": "syntax_monitor", "internal_retry_count": current_retries}
    
    recent_failures = [m for m in state.memory[-5:] if m.get('type') == 'tool_error']
    for rf in recent_failures:
        if rf.get('tool_call') == tool_draft:
            return {"rejection_feedback": "You already tried this exact action and it failed. Formulate a NEW strategy.", "rejection_source": "syntax_monitor", "internal_retry_count": current_retries}

    return {"node_logs": [{"node": "syntax_monitor", "status": "passed"}], "internal_retry_count": 0}

def validator_node(state: PevState) -> Dict:
    llm = get_llm()
    reasoning_mode = os.environ.get("AGENT_REASONING_MODE", "fc")
    current_retries = state.internal_retry_count + 1
    
    # If a fallback tool_draft was generated by syntax_monitor, IMMEDIATELY exit to Simulator
    if state.drafted_tool_call and state.drafted_tool_call.get("name") in ["respond", "transfer_to_human_agents"] and current_retries >= 5:
        return {"internal_retry_count": 0, "node_logs": [{"node": "validator", "status": "bypassed_for_fallback"}]}
        
    if current_retries >= 5:
        fallback = {"name": "respond", "arguments": {"content": "I encountered a policy violation I couldn't resolve. Let's try a different request."}}
        return {"drafted_tool_call": fallback, "internal_retry_count": 0, "rejection_feedback": None, "rejection_source": None}
        
    sys_prompt = f"""You are the VALIDATOR.
Review the following drafted tool call for logic and preconditions.
If it violates policy, use `declare_verdict` with REJECT and state the reason.
If it is safe and logically sound, use `declare_verdict` with APPROVE.

CRITICAL POLICY 1: If the Executor drafted `transfer_to_human_agents` because the user forgot their reservation_id, user_id, or other details, you MUST REJECT it. The correct action is to `respond` to ask the user.

DRAFTED TOOL:
{json.dumps(state.drafted_tool_call, indent=2)}

PRIOR MEMORY (to check preconditions):
{format_memory(state.memory)}
"""
    tools = [{
        "type": "function",
        "function": {
            "name": "declare_verdict",
            "description": "Approve or reject the drafted tool call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["APPROVE", "REJECT"]},
                    "reason": {"type": "string"}
                },
                "required": ["decision"]
            }
        }
    }]

    parsed_json, raw_log = invoke_with_paradigm(llm, sys_prompt, [], tools, reasoning_mode, "Validator")
    
    if parsed_json and parsed_json.get("name") == "declare_verdict":
        args = parsed_json.get("arguments", {})
        if args.get("decision") == "REJECT":
            return {"rejection_feedback": args.get("reason", "Rejected logically"), "rejection_source": "validator", "internal_retry_count": current_retries}
    
    return {"node_logs": [{"node": "validator", "status": "approved"}], "internal_retry_count": 0}
