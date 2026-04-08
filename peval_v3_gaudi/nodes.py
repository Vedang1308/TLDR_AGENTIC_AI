import os
import re
import json
import operator
from typing import Dict, Any, List, Optional
from peval_v4_lite.src.core.model_client import ModelClient
from .state import PevState

def get_llm():
    # Phase 3 Gaudi-Native uses the standard ModelClient mode
    return ModelClient(mode="agent")

def format_memory(memory_list: List[Dict]) -> str:
    """Converts the raw JSON memory array into a clean markdown trace."""
    if not memory_list:
        return "No prior history."
    out = []
    for i, m in enumerate(memory_list):
        if m.get('type') == 'tool_result':
            out.append(f"--- Step {i+1} ---")
            out.append(f"Action: {m.get('action_taken')}")
            out.append(f"Arguments: {json.dumps(m.get('arguments_used', {}))}")
            out.append(f"Result Observation: {str(m.get('api_observation', 'None'))}")
        elif m.get('type') == 'tool_error':
            out.append(f"--- Step {i+1} [FAILED] ---")
            out.append(f"Attempted Action: {m.get('action_taken')}")
            out.append(f"Error: {m.get('api_observation', 'None')}")
        elif m.get('action') == 'AUTO_PREFETCH':
             out.append(f"--- Step {i+1} [AUTO-PREFETCH] ---")
             out.append(f"Action: {m.get('action')}")
             out.append(f"Args: {json.dumps(m.get('args'))}")
             out.append(f"Observation: {m.get('observation')}")
    return "\n".join(out) if out else "No parseable actions."

def invoke_with_paradigm(client, sys_prompt: str, user_msgs: List, tools: List[Dict], reasoning_mode: str, role_name: str):
    """
    Standard invoker from Phase 3, adapted for ModelClient.
    """
    full_user_content = ""
    for m in user_msgs:
        # PevState uses strings/dicts, user_msgs might be LangChain messages or strings
        if hasattr(m, 'content'):
            full_user_content += f"\n{m.content}"
        elif isinstance(m, dict):
             full_user_content += f"\n{m.get('content', '')}"
        else:
            full_user_content += f"\n{str(m)}"

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

CRITICAL: The Action must be perfectly valid JSON with no trailing commas.
</tool_instructions>
"""
        final_prompt = f"{sys_prompt}\n{instruction}\n<available_tools>\n{json.dumps(tools, indent=2)}\n</available_tools>\nUser Trace: {full_user_content}"
        content = client.generate(final_prompt, stop=["Observation:", "OBSERVATION:"])
        
        try:
            action_split = content.split("Action:")[-1].strip()
            start = action_split.find('{')
            end = action_split.rfind('}')
            if start != -1 and end != -1:
                return json.loads(action_split[start:end+1]), content
        except:
            pass
        match = re.search(r'\{[^{}]*\"name\"[^{}]*\}', content)
        if match:
            try: return json.loads(match.group(0)), content
            except: pass
        return None, content

    elif reasoning_mode == "act":
        instruction = f"""
<tool_instructions>
You MUST output your decision as a raw Action block. Do NOT write any conversational text.
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}
</tool_instructions>
"""
        final_prompt = f"{sys_prompt}\n{instruction}\n<available_tools>\n{json.dumps(tools, indent=2)}\n</available_tools>\nUser Trace: {full_user_content}"
        content = client.generate(final_prompt, stop=["Observation:", "OBSERVATION:"])
        try:
            action_split = content.split("Action:")[-1].strip()
            start = action_split.find('{')
            end = action_split.rfind('}')
            if start != -1 and end != -1:
                return json.loads(action_split[start:end+1]), content
        except: pass
        return None, content

    elif reasoning_mode == "irma":
        # Iterative Reasoning and Model Alignment (IRMA) Mode
        # Focuses on step-by-step extraction and canonical alignment
        instruction = """
<tool_instructions>
You are the IRMA Logic Engine. Your goal is to align the current instruction with the exact Schema of Available Tools.
1. Identify the core intent.
2. Search for the tool that matches this intent exactly.
3. Map all required parameters from the situation history.
4. Output only the Verified JSON Action block.

Action:
{"name": "...", "arguments": {...}}
</tool_instructions>
"""
        final_prompt = f"{sys_prompt}\n{instruction}\n<available_tools>\n{json.dumps(tools, indent=2)}\n</available_tools>\nUser Trace: {full_user_content}"
        content = client.generate(final_prompt, stop=["Observation:", "OBSERVATION:"])
        try:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                return json.loads(content[start:end+1]), content
        except: pass
        return None, content

    elif reasoning_mode == "reflection":
        # Self-Reflection Mode: Forces intermediate reasoning to catch hallucinations
        instruction = """
<tool_instructions>
You are a Self-Reflective Agent. 
Step 1: Analyze the current situation in a [Reasoning] block.
Step 2: Self-verify the name and parameters of your intended tool against the list.
Step 3: Output the tool call in a JSON block.

[Reasoning]
<Your internal verification of parameter validity>

Action:
{"name": "...", "arguments": {...}}
</tool_instructions>
"""
        final_prompt = f"{sys_prompt}\n{instruction}\n<available_tools>\n{json.dumps(tools, indent=2)}\n</available_tools>\nUser Trace: {full_user_content}"
        content = client.generate(final_prompt, stop=["Observation:", "OBSERVATION:"])
        try:
            action_split = content.split("Action:")[-1].strip()
            start = action_split.find('{')
            end = action_split.rfind('}')
            if start != -1 and end != -1:
                return json.loads(action_split[start:end+1]), content
        except: pass
        return None, content

    else: # "fc"
        # We manually simulate FC for Qwen2.5/3 on Gaudi if native FC isn't enabled
        instruction = """
<tool_instructions>
Output a raw JSON dictionary representing a single tool call. 
Format: {"name": "tool_name", "arguments": {"arg1": "val1"}}
Do NOT write any text. Just output the JSON.
</tool_instructions>
"""
        final_prompt = f"{sys_prompt}\n{instruction}\n<available_tools>\n{json.dumps(tools, indent=2)}\n</available_tools>\nUser Trace: {full_user_content}"
        content = client.generate(final_prompt, stop=["Observation:", "OBSERVATION:"])
        try:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                parsed = json.loads(content[start:end+1])
                if "name" in parsed: return parsed, content
        except: pass
        return None, content

def load_live_wisdom(state: PevState):
    wisdom_file = "results/phase3/persistent_wisdom.json"
    if os.path.exists(wisdom_file):
        try:
            with open(wisdom_file, "r") as f:
                live_wisdom = json.load(f)
                state.global_wisdom = list(dict.fromkeys(state.global_wisdom + live_wisdom))
        except: pass

def planner_node(state: PevState) -> Dict:
    client = get_llm()
    reasoning_mode = os.environ.get("AGENT_REASONING_MODE", "fc")
    
    failure_history = ""
    if state.failure_log:
        lines = []
        for i, f in enumerate(state.failure_log[-6:]):
            lines.append(f"Failure {i+1}: Tried `{f.get('action')}` -> Error: {f.get('error', '?')}")
            if f.get('reflection'): lines.append(f"  Diagnosis: {f.get('reflection')}")
        failure_history = "\n".join(lines)
    else: failure_history = "None."
    
    load_live_wisdom(state)
    wisdom_section = ""
    if state.global_wisdom:
        wisdom_section = "\nGLOBAL EXPERTISE:\n" + "\n".join([f"- {w}" for w in state.global_wisdom[-15:]])

    tool_wiki_section = f"\nTECHNICAL TOOL WIKI:\n{state.tools_wiki}\n" if state.tools_wiki else ""
    reflection_section = f"\n\nERROR REFLECTION (from Learning Node):\n{state.error_reflection}\n" if state.error_reflection else ""

    # STRATEGIC KERNEL integration (Step 3/4 in Diagram)
    kernel_section = f"\nSTRATEGIC KERNEL (Compressed Context):\n{state.strategic_kernel}\n" if state.strategic_kernel else ""
    snapshot_section = f"\nWORLD SNAPSHOT (Harvested Facts):\n{json.dumps(state.world_snapshot, indent=2)}\n" if state.world_snapshot else ""

    sys_prompt = f"""You are the HIERARCHICAL STRATEGIST (Planner). 
Your ONLY tool is `submit_plan`. 

### YOUR STRATEGIC CONTEXT:
{kernel_section}
{snapshot_section}

### TECHNICAL CONSTRAINTS:
{wisdom_section}
{tool_wiki_section}
{reflection_section}

MEMORY KERNEL (Recent raw steps):
{format_memory(state.memory[-5:])}

FAILURE HISTORY:
{failure_history}

REJECTION FEEDBACK:
{f"Source: {state.rejection_source} | Message: {state.rejection_feedback}" if state.rejection_feedback else "None."}

MANDATORY POLICY CHECKLIST:
1. USER IDENTIFIED? [{"X" if state.user_identified else " "}] 
   - IF NO: You MUST prioritize obtaining user_id or reservation_id immediately.
   - IF YES: Proceed with requested services.
2. LAST ATTEMPT? {state.memory[-1].get("action_taken") if state.memory else "None."}
   - IF [] or REJECTED: Choose a DIFFERENT tool or different arguments.
"""
    tools = [{
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": "Submit a 1-2 sentence plan",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {"type": "string"},
                    "task_completed": {"type": "boolean"}
                },
                "required": ["plan", "task_completed"]
            }
        }
    }]
    
    parsed, raw = invoke_with_paradigm(client, sys_prompt, state.user_conversation[-6:], tools, reasoning_mode, "Planner")
    
    if parsed:
        name = parsed.get("name")
        args = parsed.get("arguments", {})
        if name == "submit_plan":
            plan = args.get("plan", "Proceed.")
            if args.get("task_completed", False):
                return {"task_completed": True, "error_reflection": None, "node_logs": [{"node": "planner", "plan": plan, "log": raw}]}
        else:
            plan = f"Strategic Recovery for hallucinated {name}."
        
        return {
            "current_plan": plan, 
            "rejection_feedback": None,
            "rejection_source": None,
            "error_reflection": None,
            "node_logs": [{"node": "planner", "plan": plan, "log": raw}]
        }
    return {"current_plan": "Fallback Strategy.", "node_logs": [{"node": "planner", "error": "No Strategic Output", "log": raw}]}

def translator_node(state: PevState) -> Dict:
    """
    Component: Semantic Translator (Step 5/6 in Diagram)
    Role: Normalizes the Executor's draft into a 'Verified Schema'.
    """
    draft = state.drafted_tool_call
    if not draft:
        return {"node_logs": [{"node": "translator", "status": "no_draft_to_translate"}]}

    # If already a clean dictionary with 'name' and 'arguments', we consider it 'Normalized'
    # but we can do a secondary cleaning pass here if needed.
    normalized = draft.copy()
    if "name" in normalized and "arguments" in normalized:
        # Check for nested JSON strings inside arguments (common model error)
        if isinstance(normalized["arguments"], str):
            try: normalized["arguments"] = json.loads(normalized["arguments"])
            except: pass
        
        # Ensure name is a string
        normalized["name"] = str(normalized["name"]).strip()
        
    return {
        "drafted_tool_call": normalized,
        "node_logs": [{"node": "translator", "status": "normalized_action", "normalized": normalized}]
    }

def executor_node(state: PevState) -> Dict:
    client = get_llm()
    reasoning_mode = os.environ.get("AGENT_REASONING_MODE", "fc")
    
    failed_actions_note = ""
    if state.failure_log:
        failed_names = list(set(f.get('action') for f in state.failure_log[-4:] if f.get('action')))
        failed_actions_note = f"\n\n[ALREADY FAILED]: {failed_names}\n"
    
    sys_prompt = f"""You are the EXECUTOR.
PLAN: {state.current_plan}
MEMORY: {format_memory(state.memory)}
{failed_actions_note}
Draft the single best tool call."""

    if state.rejection_feedback:
        sys_prompt += f"\n\n[REJECTION FEEDBACK from {state.rejection_source}]: {state.rejection_feedback}\n"
        if "loop" in str(state.rejection_feedback).lower() or "redundant" in str(state.rejection_feedback).lower():
            sys_prompt += "\nGUIDANCE: Do NOT repeat the previous action. Switch to a different tool (Interacting with User or searching differently).\n"

    tools = state.tools_info.copy()
    parsed, raw = invoke_with_paradigm(client, sys_prompt, [], tools, reasoning_mode, "Executor")
    return {"drafted_tool_call": parsed, "node_logs": [{"node": "executor", "raw_output": raw}]}

def syntax_monitor_node(state: PevState) -> Dict:
    draft = state.drafted_tool_call
    retries = state.internal_retry_count + 1
    
    if retries >= 5:
        return {"drafted_tool_call": {"name": "respond", "arguments": {"content": "Internal Error."}}, "internal_retry_count": 0}
    
    if not draft or "name" not in draft or "arguments" not in draft:
        return {"rejection_feedback": "Invalid JSON structure.", "rejection_source": "syntax_monitor", "internal_retry_count": retries}
        
    return {"node_logs": [{"node": "syntax_monitor", "status": "passed"}], "internal_retry_count": 0}

def validator_node(state: PevState) -> Dict:
    client = get_llm()
    retries = state.internal_retry_count + 1
    draft = state.drafted_tool_call
    
    if retries >= 5:
        return {"drafted_tool_call": {"name": "respond", "arguments": {"content": "Validation Timeout."}}, "internal_retry_count": 0}

    if draft and draft.get("name") == "respond":
        return {"node_logs": [{"node": "validator", "status": "approved (respond)"}], "internal_retry_count": 0}

    if draft and draft.get("name") == "think":
        last_action = state.memory[-1].get("action_taken") if state.memory else None
        if last_action == "think":
            return {
                "rejection_feedback": "Consecutive thinking detected without progress. Use a primary tool or respond to the user.",
                "rejection_source": "validator",
                "internal_retry_count": retries,
                "node_logs": [{"node": "validator", "status": "rejected (consecutive think)"}]
            }
            
    # Redundancy Check: Prevent the agent from repeating empty searches
    if draft and state.memory:
        for m in state.memory:
            if m.get("type") == "tool_result" and m.get("action_taken") == draft.get("name"):
                if m.get("arguments_used") == draft.get("arguments"):
                    obs = str(m.get("api_observation"))
                    if obs == "[]" or obs == "{}" or "not found" in obs.lower():
                        return {
                            "rejection_feedback": f"Redundant action. You already tried {draft.get('name')} with these arguments and it returned no results. Try a different date, city, or ask the user for more info.",
                            "rejection_source": "validator",
                            "internal_retry_count": retries,
                            "node_logs": [{"node": "validator", "status": "rejected (redundant empty search)"}]
                        }

    sys_prompt = f"""You are the VALIDATOR. Pre-flight simulate:
DRAFT: {json.dumps(state.drafted_tool_call)}
MEMORY: {format_memory(state.memory)}

Approve or Reject."""
    
    tools = [{
        "type": "function",
        "function": {
            "name": "declare_verdict",
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
    
    parsed, raw = invoke_with_paradigm(client, sys_prompt, [], tools, "fc", "Validator")
    if parsed and parsed.get("arguments", {}).get("decision") == "REJECT":
        return {
            "rejection_feedback": parsed["arguments"].get("reason", "Rejected."),
            "rejection_source": "validator",
            "internal_retry_count": retries,
            "node_logs": [{"node": "validator", "status": "rejected"}]
        }
    return {"node_logs": [{"node": "validator", "status": "approved"}], "internal_retry_count": 0}

def error_reflection_node(state: PevState) -> Dict:
    client = get_llm()
    recent_errors = [m for m in state.memory if m.get('type') == 'tool_error'][-3:]
    if not recent_errors: return {}
    
    error_summary = "\n".join([f"Action: {e.get('action_taken')} | Error: {e.get('api_observation')}" for e in recent_errors])
    sys_prompt = f"Failed states: {error_summary}\nDiagnose root cause and corrective plan."
    
    reflection_text = client.generate(sys_prompt)
    failure_entries = [{"action": e.get('action_taken'), "args": e.get('arguments_used'), "error": e.get('api_observation'), "reflection": reflection_text} for e in recent_errors]
    
    return {
        "error_reflection": reflection_text,
        "failure_log": failure_entries,
        "consecutive_error_count": 0,
        "node_logs": [{"node": "error_reflection", "reflection": reflection_text}]
    }

def reformulator_node(state: PevState) -> Dict:
    """
    Component: Input Reformulator (IRMA Node)
    Role: Transforms raw environmental observations into structured, canonical inputs.
    """
    client = get_llm()
    if not state.memory:
        return {}
        
    last_obs = state.memory[-1].get("api_observation", "")
    sys_prompt = """
You are the IRMA Input Reformulator. Your task is to extract core actionable information 
from raw noisy observations. Remove conversational filler and identify key values 
(IDs, prices, status). Output the reformulated canonical input only.
"""
    reformulated = client.generate(f"{sys_prompt}\n\nRaw Observation: {last_obs}")
    
    return {
        "reformulated_observation": reformulated,
        "node_logs": [{"node": "reformulator", "status": "reformatted", "output": reformulated}]
    }

def reflection_strategy_node(state: PevState) -> Dict:
    """
    Component: Self-Reflection Strategy Node (Reflection Routing)
    Role: Diagnoses why a tool draft was rejected and provides corrective instructions.
    """
    client = get_llm()
    draft = state.drafted_tool_call
    feedback = state.rejection_feedback
    source = state.rejection_source
    
    sys_prompt = f"""
You are the Pre-Flight Reflector. The previous tool draft was REJECTED during validation.
REJECTION SOURCE: {source}
REJECTION FEEDBACK: {feedback}
FAILED DRAFT: {json.dumps(draft)}

Diagnose the root cause (hallucination, missing parameter, schema mismatch) 
and provide a specific corrective instruction for the next planning turn.
"""
    diagnosis = client.generate(sys_prompt)
    
    return {
        "error_reflection": diagnosis, # Inject into the same slot for Planner consumption
        "node_logs": [{"node": "reflection_strategy", "diagnosis": diagnosis}]
    }

# --- ALIASES FOR LEGACY ENGINE COMPATIBILITY ---
global_reflector_node = error_reflection_node

def proactive_prefetch(env, state: PevState):
    """
    Standalone pre-fetch logic for legacy PEVEngine.
    Proactively probes for User, Reservation, or Order details (Domain-Aware).
    """
    obs = state.user_conversation[-1]["content"] if state.user_conversation else ""
    
    # 1. User ID Seeding (Common to both domains)
    user_id_match = re.search(r'\b([a-z]+_[a-z]+_\d{3,6})\b', obs, re.IGNORECASE)
    if user_id_match:
        uid = user_id_match.group(1).lower()
        tool = next((t for t in state.tools_info if 'user' in t.get('name','').lower() and 'detail' in t.get('name','').lower()), None)
        if tool:
            from tau_bench.types import Action
            res = env.step(Action(name=tool['name'], kwargs={"user_id": uid}))
            state.memory.append({"action": "AUTO_PREFETCH", "args": {"user_id": uid}, "observation": str(res.observation)})

    # 2. Reservation/Order ID Seeding (Domain-Aware Regex)
    # Airline: 6-char Alpha (e.g. ABCDEF) or alphanumeric
    # Retail: Sometimes numeric or alphanumeric
    res_id_match = re.search(r'\b([A-Z\d]{6})\b', obs)
    if res_id_match:
        rid = res_id_match.group(1)
        # Search for domain-agnostic detail tools (reservation or order)
        tool = next((t for t in state.tools_info if (
            'reservation' in t.get('name','').lower() or 
            'order' in t.get('name','').lower()
        ) and 'detail' in t.get('name','').lower()), None)
        
        if tool:
            from tau_bench.types import Action
            # Dynamically determine argument name based on tool schema
            arg_name = "reservation_id" if "reservation" in tool['name'] else "order_id"
            try:
                res = env.step(Action(name=tool['name'], kwargs={arg_name: rid}))
                state.memory.append({"action": "AUTO_PREFETCH", "args": {arg_name: rid}, "observation": str(res.observation)})
            except: pass
