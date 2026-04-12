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

def detect_progress_stagnation(state: PevState, window_size: int = 8) -> Dict[str, Any]:
    """
    Architectural Gate: Checks if the agent has made technical progress in the last N entries.
    Progress = Successful technical tool result (non-empty search, valid ID retrieval).
    Stagnation = Apologies, failed validation, or empty searches.
    """
    if not state.memory:
        return {"stagnated": False, "ratio": 0.0}
    
    window = state.memory[-window_size:]
    stagnant_entries = 0
    total = len(window)
    
    for m in window:
        # Responds and failed validations are signs of stagnation
        if m.get("action_taken") == "respond" or m.get("type") == "tool_error":
            stagnant_entries += 1
        # Empty searches are also non-progress
        elif m.get("type") == "tool_result" and "[]" in str(m.get("api_observation")):
            stagnant_entries += 1
            
    ratio = stagnant_entries / total if total > 0 else 0.0
    # If more than 75% of the window is stagnation, we hit the gate
    return {"stagnated": ratio >= 0.75, "ratio": ratio}

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
    exhausted_searches = state.world_snapshot.get("exhausted_searches", [])
    
    for m in state.memory:
        obs_str = str(m.get("api_observation", "")).strip()
        if m.get("type") == "tool_result" and "search" in str(m.get("action_taken")).lower():
            if obs_str == "[]" or obs_str == "" or obs_str.lower() == "none":
                args = m.get("arguments_used") or {}
                raw_date = str(args.get("date", ""))
                ym = re.search(r'\b(20[2-3][0-9])\b', raw_date)
                if ym:
                    y = ym.group(1)
                    year_failures[y] = year_failures.get(y, 0) + 1
    
    for y, count in year_failures.items():
        if count >= 2:
            temporal_hint = f"\n[STRATEGIC NOTE]: You have {count} failed searches for the year {y}. This often indicates a 'Year Mismatch'. Verify if the mission should be a different year (e.g. {int(y)+1}).\n"

    if exhausted_searches:
        temporal_hint += "\n[EXHAUSTED SEARCHES - PERMANENT LOG]:\n - " + "\n - ".join(exhausted_searches) + "\n"

    # ANTI-APOLOGY GATE: Prevent infinite conversational loops using Window-Based Progress
    stagnation_report = detect_progress_stagnation(state)
    stagnated = stagnation_report["stagnated"]
    
    anti_apology_mandate = ""
    hard_forbidden_respond = False
    if stagnated:
         hard_forbidden_respond = True
         anti_apology_mandate = f"\n### [!] HARD CONSTRAINT: PROGRESS STAGNATION DETECTED (Ratio: {stagnation_report['ratio']:.2f}) [!]\n"
         anti_apology_mandate += "The 'respond' tool is TEMPORARILY DISABLED. You MUST attempt a technical action with DIFFERENT parameters to break this loop.\n"

    # STRATEGIC RECOVERY HINTS (Domain Agnostic Phase 4.29)
    recovery_hints = ""
    if any("EMPTY" in str(s) or "0 results" in str(s) for s in state.tool_audit_log[-5:]):
        recovery_hints = "\n### STRATEGIC RECOVERY HINTS (Universal) ###\n"
        recovery_hints += "- If a search returns no results, try exploring alternative nearby location codes or adjacent dates (+/- 1 day).\n"
        recovery_hints += "- If identifying the user fails, prioritize verifying the spelling of names or trying a different identification tool.\n"
        recovery_hints += "- You MUST change at least one mandatory parameter to explore new solution space.\n"

    # TOOL AUDIT LOG formatting (User Suggestion)
    audit_section = "\n### TECHNICAL TOOL AUDIT LOG (Internal Technical History) ###\n"
    if state.tool_audit_log:
        audit_section += "\n".join(state.tool_audit_log[-12:]) + "\n"
    else:
        audit_section += "No technical attempts yet.\n"

    # STRATEGIC KERNEL integration
    kernel_section = f"\nSTRATEGIC KERNEL (Compressed Context):\n{state.strategic_kernel}\n" if state.strategic_kernel else ""
    snapshot_section = f"\n### VERIFIED TECHNICAL SCRATCHPAD (WORLD SNAPSHOT) ###\n{json.dumps(state.world_snapshot, indent=2) if state.world_snapshot else '[] - DISCOVERY REQUIRED'}\n{temporal_hint}\n{audit_section}\n{recovery_hints}\n{anti_apology_mandate}\n"

    # TOOL LOCK LOGIC
    discovery_tools = ["get_user_details", "get_reservation_details", "list_reservations"]
    mutation_tools = ["book_reservation", "update_reservation_flights", "update_reservation_baggages", "cancel_reservation"]
    
    has_res_id = any("reservation_id" in str(k) or "available_reservations" in str(k) for k in state.world_snapshot.keys())
    lock_status = "\n### TOOL LOCK STATUS ###\n"
    lock_status += f"- DISCOVERY TOOLS: UNLOCKED (Always available)\n"
    if has_res_id:
        lock_status += f"- MUTATION TOOLS: UNLOCKED (Reservation ID found in Scratchpad)\n"
    else:
        lock_status += f"- MUTATION TOOLS: [LOCKED] (You MUST call a Discovery tool first to find a reservation_id)\n"

    # VICTORY DETECTION
    victory_status = ""
    if state.world_snapshot and any(k in str(v).lower() for k in ["reservation_id", "status_confirmed", "cancelled"] for v in state.world_snapshot.values()):
        victory_status = "\n[!] VICTORY DETECTED: Technical win confirmed. STOP technical work. END the trial using 'respond' immediately.\n"

    sys_prompt = f"""You are the HIERARCHICAL STRATEGIST (Planner). 
Your ONLY tool is `submit_plan`. 

### IMMOVABLE MISSION (ORIGINAL REQUEST) ###
{initial_mission}
{victory_status}
{"[!!!] FORBIDDEN ACTION: DO NOT USE THE 'respond' TOOL THIS TURN. YOU ARE IN A CONVERSATIONAL LOOP. [!!!]" if hard_forbidden_respond else ""}

{snapshot_section}
{lock_status}

### THE DYNAMIC REQUIREMENT CHECKLIST ###
You must track EVERY part of the user's request. 
Example check (if applicable):
- [ ] User Identification (get_user_details)
- [ ] Flight Discovery (search_direct_flight / search_onestop_flight)
- [ ] Primary Action (book_reservation / update_reservation_flights)
- [ ] Ancillary Actions (update_reservation_baggages / update_reservation_insurance)
- [ ] Final Confirmation (respond)

### MANDATORY POLICIES ###
1. THE IDENTITY GROUNDING MANDATE: 
   - Your absolute first priority is to call `get_user_details`. 
   - DO NOT attempt to modify reservations or search specifically for a user's existing trip until they are grounded.

2. THE SCRATCHPAD MANDATE:
   - You are STRICTLY FORBIDDEN from using an ID (Reservation ID, Payment ID, Flight Number) that does not appear in the VERIFIED TECHNICAL SCRATCHPAD.
   - If a tool requires an ID you don't have, your ONLY option is to use a Discovery tool or ASK the user. NEVER guess.

3. THE EXHAUSTION MANDATE:
   - If a search tool returns `[]`, mark that date/route as EXHAUSTED. 
   - Pivot immediately to adjacent dates or alternative airport codes. Do NOT repeat failed searches.

4. TRANSACTIONAL INTEGRITY & NUMERICAL OWNERSHIP:
   - For multi-part requests (e.g. flight change + baggage), you MUST perform the flight change first, verify the new reservation state, then perform the baggage update. DO NOT skip the ancillary parts.
   - If a tool returns a 'Payment Mismatch' or 'Does not add up' error, your next step MUST be to re-verify all components (Base Price + Ancillary Fees) against the VERIFIED TECHNICAL SCRATCHPAD. Use the `calculate` tool if the math is complex.

5. THE GUIDANCE MANDATE:
   - Your task is NOT successfully completed until you have explicitly addressed every 'How-to', 'Insurance Claim', or 'Procedure' request from the user. 
   - Simply performing the technical action (e.g., cancel) is a partial failure if the user asked for guidance. You MUST explain the process, refund rules, or insurance steps in your final response.

6. THE COMPLETION MANDATE (MANDATORY):
   - You only succeed if you CONFIRM the outcome to the user. After every final state-mutation (book, update, cancel), you MUST use the `respond` tool to tell the user exactly what was done and what the final state is.
   - DO NOT just stop after a technical tool call. The loop is only closed when you respond.

7. THE FACT-ANCHORING MANDATE:
   - You are STRICTLY FORBIDDEN from fabricating technical data. 
   - NEVER state a flight is "delayed", "on time", or has a specific "reason" (like weather) unless that EXACT text appears in a tool observation in your MEMORY.
   - If a tool for 'Flight Status' does not exist, you must inform the user you cannot check it. NEVER make up a status.

### YOUR STRATEGIC CONTEXT:
{kernel_section}

### TECHNICAL CONSTRAINTS:
{wisdom_section}
{tool_wiki_section}
{reflection_section}

MEMORY KERNEL (Recent raw steps):
{format_memory(state.memory[-15:])} # Memory window increased to prevent amnesia

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
3. CONSECUTIVE FAILURES? If any tool has > 3 attempts in 'TOOL ATTEMPT COUNTS' or 'AUDIT LOG', you MUST PIVOT autonomously. Do NOT just repeat. Instead:
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
    
    # [PHASE 4.29] ARCHITECTURAL LOOP CIRCUIT-BREAKER
    forbidden_note = ""
    stagnation_report = detect_progress_stagnation(state)
    if stagnation_report["stagnated"]:
        forbidden_note = f"\n\n### [!!!] FORBIDDEN ACTION: 'respond' IS DISABLED DUE TO STAGNATION (Ratio: {stagnation_report['ratio']:.2f}) [!!!]\n"
        forbidden_note += "YOU MUST ATTEMPT A TECHNICAL TOOL CALL OR STRATEGY SHIFT.\n"

    # Tool Audit Log and Snapshot for Executor (Phase 4.31 Fix)
    audit_section = "\n### TECHNICAL TOOL AUDIT LOG (Internal Technical History) ###\n"
    if state.tool_audit_log:
        audit_section += "\n".join(state.tool_audit_log[-12:]) + "\n"
    else:
        audit_section += "No technical attempts yet.\n"
        
    snapshot_section = f"\n### VERIFIED TECHNICAL SCRATCHPAD (WORLD SNAPSHOT) ###\n{json.dumps(state.world_snapshot, indent=2) if state.world_snapshot else '[] - DISCOVERY REQUIRED'}\n"

    sys_prompt = f"""You are the TECHNICAL EXECUTOR.
PLAN: {state.current_plan}
MEMORY: {format_memory(state.memory)}
{audit_section}
{snapshot_section}
{failed_actions_note}
{forbidden_note}

### MANDATORY SCHEMA GROUNDING (GROUND TRUTH) ###
You are FORBIDDEN from drafting technical tools without ALL mandatory arguments.
Reference your Tool Wiki carefully. Common requirements:
- search_direct_flight / search_onestop_flight: REQUIRES [origin], [destination], [date]
- book_reservation: REQUIRES [user_id], [flight_numbers], [passenger_names], [payment_id]
- get_user_details: REQUIRES [user_id]
- update_reservation_flights: REQUIRES [reservation_id], [new_flights], [payment_id]

### CRITICAL DISCOVERY MANDATE ###
If the user provides a user_id or reservation_id, and it is NOT present in your Verified Technical Scratchpad, you MUST call the discovery tool ('get_user_details' or 'get_reservation_details') immediately.

Draft the single best tool call. Choose concisely."""

    if state.rejection_feedback:
        sys_prompt += f"\n\n[!!! CRITICAL CORRECTION !!! from {state.rejection_source.upper()}]:\n>> {state.rejection_feedback}\n"
        sys_prompt += "\nYour previous attempt was INVALID. You MUST change your arguments or strategy based on this feedback immediately. DO NOT repeat the same mistake.\n"
        if "loop" in str(state.rejection_feedback).lower() or "redundant" in str(state.rejection_feedback).lower() or "TIMEOUT" in str(state.rejection_feedback):
            sys_prompt += "\nGUIDANCE: Do NOT repeat the previous action. Switch to a DIFFERENT tool or check the Scratchpad for corrected IDs.\n"

    tools = state.tools_info.copy()
    
    # [PHASE 4.28/29] PHYSICAL TOOL FILTERING (The Nuclear Option)
    stagnation_report = detect_progress_stagnation(state)
    if stagnation_report["stagnated"]:
        # Physically remove the 'respond' tool from available capabilities
        tools = [t for t in tools if t.get("function", {}).get("name") != "respond"]
        sys_prompt += "\n[SYSTEM ALERT]: THE 'respond' TOOL IS DISABLED. YOU MUST PERFORM A TECHNICAL ACTION OR SEARCH VARIATION.\n"

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
        # [PHASE 4.29] WINDOW-BASED LOOP DETECTION
        stagnation_report = detect_progress_stagnation(state)
        
        # VICTORY BYPASS: If the IMMEDIATELY PRECEDING action was a technical success, allow 'respond'.
        technical_success = False
        if state.memory:
            for m in reversed(state.memory):
                if m.get("type") == "tool_result" and m.get("action_taken") != "think":
                    if m.get("action_taken") != "respond" and "Error" not in str(m.get("api_observation")) and "[]" not in str(m.get("api_observation")):
                        technical_success = True
                    break # ALWAYS break on the first non-think tool_result

        if stagnation_report["stagnated"] and not technical_success:
            return {
                "rejection_feedback": f"Infinite Loop Detected: Progress Stagnation Ratio: {stagnation_report['ratio']:.2f}. You are FORBIDDEN from responding further until you make technical progress. Call a different technical tool now and CHANGE your arguments.",
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
            retry_keywords = ["check again", "retry", "are you sure", "try again", "double check", "re-search", "mix-up", "actually", "wrong", "rephrase", "looking for", "not to", "correction", "different"]
            if any(k in last_msg for k in retry_keywords):
                user_retry_intent = True

        # 2. Airtight Fingerprinting (Fixed: Granular Argument Hashing for utility tools)
        def get_fingerprint(tool_name, args):
            if not args: return tool_name
            
            # For Search tools, stick to Origin|Destination|Date
            if "search" in tool_name.lower():
                raw_date = str(args.get('date', "")).lower().strip()
                month_map = {"jan":"01", "feb":"02", "mar":"03", "apr":"04", "may":"05", "jun":"06", 
                             "jul":"07", "aug":"08", "sep":"09", "oct":"10", "nov":"11", "dec":"12"}
                year, m_found, day = "NODATE", "NODATE", "NODATE"
                if raw_date:
                    iso_parts = re.split(r'[-/]', raw_date)
                    if len(iso_parts) >= 3:
                        year = iso_parts[0] if len(iso_parts[0]) == 4 else "2024"
                        m_found = iso_parts[1].zfill(2)
                        day = iso_parts[2].zfill(2)
                    else:
                        nums = re.findall(r'\d+', raw_date)
                        for m_name, m_val in month_map.items():
                            if m_name in raw_date:
                                m_found = m_val; break
                        if nums:
                            nums_sorted = sorted([int(n) for n in nums], reverse=True)
                            if nums_sorted[0] > 1000: year = str(nums_sorted[0])
                            if len(nums_sorted) > 1: day = str(nums_sorted[-1])
                norm_date = f"{year}{m_found}{str(day).zfill(2)}"
                return f"{tool_name.upper()}:{str(args.get('origin', '')).strip().upper()}|{str(args.get('destination', '')).strip().upper()}|{norm_date}"
            
            # For Calculation or Detail tools, use the entire argument string to detect unique work
            # This prevents collissions between two DIFFERENT calculations (e.g. Economy vs Business)
            return f"{tool_name.upper()}:{json.dumps(args, sort_keys=True)}"

        # 3. Apply Gate
        if not user_retry_intent:
            immutable_tools = [
                "search_direct_flight", "search_onestop_flight", 
                "get_user_details", "update_reservation_baggages", 
                "book_reservation", "calculate", "list_reservations", "get_reservation_details"
            ]
            current_fp = get_fingerprint(draft.get("name"), draft.get("arguments", {}))
            for m in state.memory:
                if m.get("type") in ["tool_result", "tool_error"] and m.get("action_taken") == draft.get("name"):
                    # FIXED: Only reject if the previous result was SUCCESSFUL.
                    # If it returned an "Error", the agent must be allowed to try again with fixed parameters.
                    # [PHASE 4.23]: Allow discovery/calculation retries if the previous action was a tool_error.
                    prev_obs = str(m.get("api_observation", "")).lower()
                    is_discovery = any(kw in draft.get("name") for kw in ["calculate", "get_", "list_"])
                    
                    if ("error" in prev_obs or m.get("type") == "tool_error") and is_discovery:
                        continue # Allow retry of discovery tools to fix errors
                    elif "error" in prev_obs or m.get("type") == "tool_error":
                        continue # Allow retry after failures
                        
                    prev_args = m.get("arguments_used") or m.get("arguments") or {}
                    prev_fp = get_fingerprint(m.get("action_taken"), prev_args)
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

    # 2. HALLUCINATION BLOCKADE (Scratchpad-Integrated)
    # Ensure any ID used in a state-mutation tool (book, update, delete) exists in the World Snapshot.
    if draft and any(k in draft.get("name", "").lower() for k in ["book", "update", "delete", "cancel"]):
        snapshot_str = str(state.world_snapshot)
        history_str = str(state.user_conversation)
        
        for key, value in draft.get("arguments", {}).items():
            val_str = str(value)
            # Match typical system ID patterns
            if re.search(r'^[a-zA-Z0-9]+_[a-zA-Z0-9]{4,}$', val_str):
                # Valid if found in Snapshot OR provided by User
                if val_str not in snapshot_str and val_str not in history_str:
                    return {
                        "rejection_feedback": f"ID Hallucination Blockade: The identifier '{val_str}' is NOT in the Verified Technical Scratchpad. You are FORBIDDEN from using IDs that haven't been harvested by a tool call or given by the user. Use a Discovery tool (get_user_details, etc.) first.",
                        "rejection_source": "validator",
                        "internal_retry_count": retries,
                        "node_logs": [{"node": "validator", "status": "rejected (scratchpad violation)"}]
                    }

    sys_prompt = f"""You are the strict structural and TECHNICAL VALIDATOR. 
DRAFTED ACTION: {json.dumps(state.drafted_tool_call)}
HISTORY: {format_memory(state.memory)}

### THE GROUND TRUTH: AVAILABLE TOOLS WIKI ###
{json.dumps(state.tools_info, indent=2)}

### YOUR MANDATE ###
1. SCHEMA-FIRST VALIDITY: Verify tool name and argument types against the Ground Truth Wiki above.
   - NEVER suggest or allow a tool name that is not in the Wiki. If the agent drafts a non-existent tool (e.g., 'search_multistop_flight'), you MUST reject it.
   - Do NOT try to be 'helpful' by guessing what tool should exist. If it is not in the Wiki, it does NOT exist.
2. CAPACITY & ID VERIFICATION (STRICT):
   - For booking/update actions: SUM all amounts in any payment lists. Verify against prices in MEMORY.
   - VALID IDs: Approve identifiers if they appeared in EITHER a technical tool result (MEMORY) OR were provided by the user in the chat (HISTORY). 
   - REJECT if the agent tries to book an item previously seen as having 0 or NONE availability/stock/seats. 
3. SEMANTIC FLUIDITY (SOURCE OF TRUTH):
   - The User is the Source of Truth. If the user corrects a route, date, or preference in a message, APPROVE IT. 
   - VICTORY BYPASS: If the VERY LAST TECHNICAL ACTION in MEMORY (ignore previous 'think' or 'respond' blocks) was a SUCCESSFUL state-mutation tool (e.g. update, book, calculate, or cancel), you MUST APPROVE the drafting of a final 'respond' call.
   - DO NOT enforce tool "consistency." Pivoting between tools is a sign of intelligence.
4. FALSE REJECTIONS ARE FATAL (ANTI-HALLUCINATION):
   - YOU MUST NOT REJECT an action by claiming it was "already attempted" unless you see the EXACT SAME ARGUMENTS in the HISTORY. Retrying a tool with DIFFERENT arguments, DIFFERENT dates, or DIFFERENT airport codes is VALID EXPLORATION and MUST be APPROVED.
   - NEVER reject a tool by claiming it "does not exist in the environment" if it is listed in the Ground Truth Wiki. Read the function names carefully.

Your job is NOT to judge high-level strategy. Verify strictly against the Wiki and physical constraints.
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
