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
    """Converts the raw JSON memory array into a clean markdown trace with aggressive truncation."""
    if not memory_list:
        return "No prior history."
    out = []
    for i, m in enumerate(memory_list):
        obs = str(m.get('api_observation', 'None'))
        if len(obs) > 800:
            obs = obs[:800] + "... [TRUNCATED for brevity]"
            
        if m.get('type') == 'tool_result':
            out.append(f"--- Step {i+1} ---")
            out.append(f"Action: {m.get('action_taken')}")
            out.append(f"Arguments: {json.dumps(m.get('arguments_used', {}))}")
            out.append(f"Result Observation: {obs}")
        elif m.get('type') == 'tool_error':
            out.append(f"--- Step {i+1} [FAILED] ---")
            out.append(f"Attempted Action: {m.get('action_taken')}")
            out.append(f"Error: {obs}")
        elif m.get('action') == 'AUTO_PREFETCH':
             out.append(f"--- Step {i+1} [AUTO-PREFETCH] ---")
             out.append(f"Action: {m.get('action')}")
             out.append(f"Args: {json.dumps(m.get('args'))}")
             out.append(f"Observation: {obs}")
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
<Evidence-based reasoning citing specific steps from your MEMORY to map required arguments.>
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

    # MISSION ANCHOR: Search history for the first message containing actual data (City, Date, or Year)
    initial_mission = "Unknown Mission"
    # DYNAMIC YEAR DETECTION: Identify current operating year from context
    working_year = "2024" # Default fallback
    for msg in state.user_conversation:
        content = msg.get('content', '')
        year_match = re.search(r'\b(20[2-3][0-9])\b', content)
        if year_match: working_year = year_match.group(1)
        if initial_mission == "Unknown Mission" and (year_match or re.search(r'\w+ to \w+', content)):
            initial_mission = content

    if initial_mission == "Unknown Mission" and state.user_conversation:
        initial_mission = state.user_conversation[0].get('content', 'Unknown Mission')

    # MISSION ANCHOR (Temporal Support): Detect repeat empty searches by year
    temporal_hint = ""
    year_failures = {}
    for m in state.memory:
        if m.get("type") == "tool_result" and "search" in str(m.get("action_taken")).lower():
            if str(m.get("api_observation")) == "[]":
                args = m.get("arguments_used") or {}
                raw_date = str(args.get("date", ""))
                ym = re.search(r'\b(20[2-3][0-9])\b', raw_date)
                if ym:
                    y = ym.group(1)
                    year_failures[y] = year_failures.get(y, 0) + 1
    
    for y, count in year_failures.items():
        if count >= 2:
            temporal_hint = f"\n[STRATEGIC NOTE]: You have {count} failed searches for the year {y}. This often indicates a 'Year Mismatch'. Verify if the mission should be a different year (e.g. {int(y)+1}).\n"

    # ANTI-APOLOGY GATE: Prevent infinite conversational loops
    respond_count = 0
    for m in reversed(state.memory):
        if m.get("type") == "action" and m.get("action_taken") == "respond":
            respond_count += 1
        else: break
    
    anti_apology_mandate = ""
    if respond_count >= 2:
        anti_apology_mandate = f"\n[TACTICAL PIVOT MANDATE]: You have responded to the user {respond_count} times in a row without technical progress. STOP apologizing or summarizing. You MUST attempt a technical tool call (search, get_details, etc.) NOW to break this loop.\n"

    # STRATEGIC KERNEL integration
    kernel_section = f"\nSTRATEGIC KERNEL (Compressed Context):\n{state.strategic_kernel}\n" if state.strategic_kernel else ""
    snapshot_section = f"\nWORLD SNAPSHOT (Harvested Facts):\n{json.dumps(state.world_snapshot, indent=2)}\n{temporal_hint}\n{anti_apology_mandate}\n" if state.world_snapshot else f"{temporal_hint}\n{anti_apology_mandate}\n"

    # VICTORY DETECTION: Check for existing reservation/status success
    victory_status = ""
    if state.world_snapshot and any(k in str(v).lower() for k in ["reservation_id", "booking_id", "order_id", "status_confirmed"] for v in state.world_snapshot.values()):
        victory_status = "\n[!] VICTORY DETECTED: A successful confirmation ID already exists in your snapshot. Your goal is now to confirm details to the user and FINISH.\n"

    sys_prompt = f"""You are the HIERARCHICAL STRATEGIST (Planner). 
Your ONLY tool is `submit_plan`. 

### IMMOVABLE MISSION (ORIGINAL REQUEST) ###
{initial_mission}
{victory_status}

### THE EXHAUSTION MANDATE (CRITICAL):
- If a search tool returns an empty list `[]`, that means NO flights exist for those parameters.
- The environment is unforgiving. If a search tool returns a flight with `available_seats: 0` for your target class, or a departure time that violates user constraints, that flight is a DEAD END.
- DO NOT repeat the same search. DO NOT apologize endlessly.
- If a search result (like `search_direct_flight`) fails your constraints (e.g., all flights are at 6 AM but user wants after 11 AM), you MUST pivot intelligently: 
    1. IMMEDIARE FALLBACK: If direct flights fail constraints, try `search_onestop_flight` immediately.
    2. ADJACENT SEARCH: Suggest/try adjacent dates (e.g. if May 20 is empty, try May 21).
    3. ADJACENT LOCATIONS: Suggest different airport options in the same region.
- Your response to the user must be ACTIONABLE: If you have no technical tools left to try, summarize exactly what you checked and ask the user for a specific pivot (date or airport).
- MATH & PRICING: If the user asks for "total cost", "price", or "total", you MUST use the `calculate` tool. Do NOT estimate or guess totals in conversation.
- ID CONTEXT MANDATE: Whenever using a reservation-specific tool (update, cancel, get_reservation_details), you MUST include BOTH 'user_id' AND 'reservation_id' if both are available in MEMORY.
- PROACTIVE HARVESTING: If you are missing a 'reservation_id', do NOT just ask the user. Immediately call `get_user_details` to harvest any existing reservations from the user's profile.

### YOUR STRATEGIC CONTEXT:
{kernel_section}
{snapshot_section}

### TECHNICAL CONSTRAINTS:
{wisdom_section}
{tool_wiki_section}
{reflection_section}

MEMORY KERNEL (Recent raw steps):
{format_memory(state.memory[-5:])}

TOOL ATTEMPT COUNTS (Task-wide):
{json.dumps(state.tool_attempts, indent=2) if state.tool_attempts else "None."}

FAILURE HISTORY:
{failure_history}

### REJECTION HISTORY (MUST RESOLVE) ###
{f"CRITICAL ERROR FROM {state.rejection_source.upper()}: {state.rejection_feedback}" if state.rejection_feedback else "None."}
{f"PREVIOUS ATTEMPTED ACTION: {state.drafted_tool_call.get('name')}" if state.drafted_tool_call else ""}

MANDATORY POLICY CHECKLIST:
1. USER IDENTIFIED? [{"X" if state.user_identified else " "}] 
   - IF NO: You MUST prioritize obtaining user_id or reservation_id immediately.
   - IF YES: Proceed with requested services.
3. CONSECUTIVE FAILURES? If any tool has > 3 attempts in 'TOOL ATTEMPT COUNTS', you MUST PIVOT autonomously. Do NOT just repeat. Instead:
   - Try a different search tool (e.g., one-stop instead of direct).
   - Change search parameters (e.g., +/- 1 day or different airport).
   - Re-calculate and verify your arguments against the MEMORY.
   - Only ask the user for clarification as a last resort if all tool variations are exhausted.
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
        sys_prompt += f"\n\n[!!! CRITICAL CORRECTION !!! from {state.rejection_source.upper()}]:\n>> {state.rejection_feedback}\n"
        sys_prompt += "\nYour previous attempt was INVALID. You MUST change your arguments or strategy based on this feedback immediately. DO NOT repeat the same mistake.\n"
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

def normalize_args(args: Any) -> str:
    """Normalize tool arguments for robust comparison."""
    if not args: return "{}"
    if isinstance(args, str):
        try:
            val = json.loads(args)
            if isinstance(val, dict): return json.dumps(val, sort_keys=True)
        except: pass
        return str(args).strip().lower()
    if isinstance(args, dict):
        return json.dumps(args, sort_keys=True)
    return str(args).strip().lower()

def validator_node(state: PevState) -> Dict:
    client = get_llm()
    retries = state.internal_retry_count + 1
    draft = state.drafted_tool_call
    
    if retries >= 5:
        return {"drafted_tool_call": {"name": "respond", "arguments": {"content": "Validation Timeout."}}, "internal_retry_count": 0}

    if draft and draft.get("name") == "respond":
        # CONVERSATIONAL LOOP DETECTION: Prevent infinite 'apology loops'
        # Fixed: Use action_taken key to sync with Engine's memory storage
        respond_count = 0
        for m in reversed(state.memory):
            if m.get("action_taken") == "respond":
                respond_count += 1
            else: break
        
        if respond_count >= 2:
            return {
                "rejection_feedback": f"Infinite Loop Detected: You have responded to the user {respond_count} times in a row without making technical progress. You are FORBIDDEN from responding further until you perform a technical action (search, get_details, etc.). Call a tool now.",
                "rejection_source": "validator",
                "internal_retry_count": retries,
                "node_logs": [{"node": "validator", "status": "rejected (forced technical pivot)"}]
            }
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
            
    # Redundancy Check: Prevent the agent from repeating ANY search it already has data for
    # Redundancy Check: Prevent the agent from repeating technical actions it already has data for
    if draft and state.memory:
        # 1. Detect User Retry Intent
        user_retry_intent = False
        if state.user_conversation:
            last_msg = str(state.user_conversation[-1].get("content", "")).lower()
            retry_keywords = ["check again", "retry", "are you sure", "try again", "double check", "re-search"]
            if any(k in last_msg for k in retry_keywords):
                user_retry_intent = True

        # 2. Airtight Fingerprinting (Fixed: Robust Date Normalization)
        def get_fingerprint(args):
            if not args: return ""
            raw_date = str(args.get('date', "")).lower().strip()
            
            # Month mapping
            month_map = {"jan":"01", "feb":"02", "mar":"03", "apr":"04", "may":"05", "jun":"06", 
                         "jul":"07", "aug":"08", "sep":"09", "oct":"10", "nov":"11", "dec":"12"}
            
            year, m_found, day = "2024", "01", "01"
            
            # Robust ISO/Numeric Parsing (e.g. 2024-05-20)
            iso_parts = re.split(r'[-/]', raw_date)
            if len(iso_parts) >= 3:
                # Assuming YYYY-MM-DD
                year = iso_parts[0] if len(iso_parts[0]) == 4 else year
                m_found = iso_parts[1].zfill(2)
                day = iso_parts[2].zfill(2)
            else:
                # Fallback to Natural Language Parsing
                nums = re.findall(r'\d+', raw_date)
                for m_name, m_val in month_map.items():
                    if m_name in raw_date:
                        m_found = m_val; break
                if nums:
                    nums_sorted = sorted([int(n) for n in nums], reverse=True)
                    if nums_sorted[0] > 1000: year = str(nums_sorted[0])
                    if len(nums_sorted) > 1: day = str(nums_sorted[-1])

            norm_date = f"{year}{m_found}{str(day).zfill(2)}"
            return f"{str(args.get('origin', '')).strip().upper()}|{str(args.get('destination', '')).strip().upper()}|{norm_date}"

        # 3. Apply Gate
        if not user_retry_intent:
            immutable_tools = [
                "search_direct_flight", "search_onestop_flight", 
                "get_user_details", "update_reservation_baggages", 
                "book_reservation", "calculate", "list_reservations", "get_reservation_details"
            ]
            current_fp = get_fingerprint(draft.get("arguments"))
            for m in state.memory:
                if m.get("type") in ["tool_result", "tool_error"] and m.get("action_taken") == draft.get("name"):
                    prev_args = m.get("arguments_used") or m.get("arguments") or {}
                    prev_fp = get_fingerprint(prev_args)
                    if prev_fp == current_fp and draft.get("name") in immutable_tools:
                        return {
                            "rejection_feedback": f"Broad Redundancy Gate: You already performed '{draft.get('name')}' for {current_fp} at Step {state.memory.index(m)+1}. Repeating this exactly is a waste. Pivot or check results.",
                            "rejection_source": "validator",
                            "internal_retry_count": retries,
                            "node_logs": [{"node": "validator", "status": "rejected (hard redundant action)"}]
                        }

    # SEAT-AWARENESS & ID-SAFE GROUNDING (New for Phase 4.1)
    # 1. UNIVERSAL CAPACITY HEURISTICS (Domain Agnostic)
    # Scan memory for common inventory markers (seats, quantity, available, stock) paired with 0
    inventory_keys = ["seat", "quantity", "available", "stock", "inventory", "count"]
    for m in reversed(state.memory):
        obs_str = str(m.get("api_observation", "")).lower()
        draft_str = str(draft).lower()
        if any(k in obs_str for k in inventory_keys):
            # If the observation shows '0' or 'none' for a capacity key, and the draft uses that ID
            if any(f'"{k}": 0' in obs_str or f"'{k}': 0" in obs_str for k in inventory_keys):
                # We need a heuristic to see if the draft is trying to use a 'sold out' item
                # This check ensures the agent doesn't loop booking impossible items in ANY domain.
                pass # The prompt below will enforce the exact logic based on this harvested context.

    # 2. UNIVERSAL ID-SAFE GROUNDING
    # Ensure any ID used in a state-mutation tool (book, update, delete) exists in memory.
    if draft and any(k in draft.get("name", "").lower() for k in ["book", "update", "delete", "cancel"]):
        # Robust ID detection: Only flag values that look like system IDs (prefixed and numeric/random)
        # and ensure we aren't flagging Tool Schema keys (like flight_type)
        memory_str = str(state.memory)
        for key, value in draft.get("arguments", {}).items():
            val_str = str(value)
            # Match typical system ID patterns (e.g., credit_card_123, reservation_ABC)
            if re.search(r'^[a-zA-Z]+_[a-zA-Z0-9]{4,}$', val_str):
                if val_str not in memory_str:
                    return {
                        "rejection_feedback": f"ID Hallucination: The identifier '{val_str}' has not appeared in your memory. You MUST call a discovery tool (get_details, list_all, etc.) to find valid IDs before performing this action.",
                        "rejection_source": "validator",
                        "internal_retry_count": retries,
                        "node_logs": [{"node": "validator", "status": "rejected (hallucinated id)"}]
                    }

    sys_prompt = f"""You are the strict structural and TECHNICAL VALIDATOR. 
DRAFT: {json.dumps(state.drafted_tool_call)}
MEMORY: {format_memory(state.memory)}

### YOUR MANDATE ###
1. SCHEMA-FIRST VALIDITY: Verify tool name and argument types against the Tool Wiki.
   - DO NOT reject for missing user constraints (like time, baggage, or insurance) if those parameters are NOT in the Tool Wiki JSON schema for that tool.
   - If a tool only accepts 'origin', 'destination', and 'date', you MUST approve technically valid drafts even if the user has other preferences. Filtering results is the Planner's job post-tool.
2. CAPACITY & ID VERIFICATION (STRICT):
   - For booking/update actions: SUM all amounts in any payment lists. Verify against prices in MEMORY.
   - REJECT if any identifier (ID) used in the draft has NOT appeared in your MEMORY at a previous step.
   - REJECT if the agent tries to book an item previously seen as having 0 or NONE availability/stock/seats. 
3. SEMANTIC FLUIDITY (SOURCE OF TRUTH):
   - The User is the Source of Truth. If the user corrects a route, date, or preference in a message, APPROVE IT. 
   - DO NOT reject based on a previous "Initial Mission" if the User has provided new data.
   - 24-HOUR CONVERSION: 'After 11 AM' is 11:00-23:59 (19:00 is successful).
   - VICTORY BYPASS: If the VERY LAST TECHNICAL ACTION in MEMORY (ignore previous 'think' or 'respond' blocks) was a SUCCESSFUL state-mutation tool (e.g. update, book, calculate, or cancel), you MUST APPROVE the drafting of a final 'respond' call to close the loop with the user.
   - DO NOT enforce tool "consistency." Pivoting between tools is a sign of intelligence.

Your job is NOT to judge high-level strategy or path selection. Approve if technical schema and physical constraints are met.
"""
    
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
