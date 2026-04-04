from typing import Dict, Any, List, Tuple, Optional
__VERSION__ = "5.1_PHASE_STABILIZATION"
from peval_v3_gaudi.state import PevState
# Import the existing ModelClient (using absolute paths)
from peval_v4_lite.src.core.model_client import ModelClient
from peval_v4_lite.src.core.logger import PEVLogger
import math

def get_compact_tool_catalog(tools_info: List[Dict]) -> str:
    """Dynamically generates a runtime summary of available tools for the Planner/Executor (Sanitized)."""
    catalog = []
    import re
    for t in tools_info:
        name = t.get("name") or t.get("function", {}).get("name", "unknown")
        desc = t.get("description") or t.get("function", {}).get("description", "No description provided.")
        # Sanitize: Remove all "such as...", "e.g.", and example values generically
        desc = re.sub(r",? (?:such as|e\.g\.|example:).*?(\.|$)", r"\1", desc, flags=re.IGNORECASE)
        catalog.append(f"- [{name}]: {desc}")
    return "\n".join(catalog)

def format_memory(memory_list: List[Dict]) -> str:
    """Enhanced: Surfaces Success Milestones at the top for the Planner."""
    if not memory_list:
        return "No prior history."
    
    milestones = []
    out = []
    for i, m in enumerate(memory_list):
        obs = str(m.get('api_observation', ''))
        # Scrape for milestones
        res_id = re.search(r'reservation_id":\s*"([A-Z0-9]+)"', obs)
        if res_id: milestones.append(f"SUCCESS: Found Reservation ID {res_id.group(1)}")
        
        user_resolved = re.search(r'first_name":\s*"([^"]+)"', obs)
        if user_resolved and "get_user_details" in m.get('action_taken', ''):
            milestones.append(f"SUCCESS: Resolved Identity for {user_resolved.group(1)}")

        if m.get('type') == 'tool_result':
            out.append(f"--- Step {i+1} [SUCCESS] ---")
            out.append(f"Action: {m.get('action_taken')} | Args: {json.dumps(m.get('arguments_used', {}))}")
            out.append(f"Observation: {obs[:300]}...")
        elif m.get('type') == 'tool_error':
            out.append(f"--- Step {i+1} [FAILED] ---")
            out.append(f"Action: {m.get('action_taken')} | Error: {obs}")

    milestone_dashboard = "### MILESTONE DASHBOARD ###\n" + ("\n".join(set(milestones)) if milestones else "None yet.")
    return milestone_dashboard + "\n\n### DETAILED HISTORY ###\n" + "\n".join(out)

def invoke_with_paradigm(client: ModelClient, sys_prompt: str, user_msgs: List[Dict], tools: List[Dict], reasoning_mode: str) -> Tuple[Optional[Dict], str]:
    """
    Universal wrapper to force the agent to output using ReAct or strict JSON.
    Ported from Phase 3 with local ModelClient compatibility.
    """
    if reasoning_mode == "react":
        instruction = """
### TOOL INSTRUCTIONS:
You MUST follow this exact sequence:
1. Think step-by-step about what to do next inside a 'Thought:' block.
2. Output a valid JSON 'Action:' block containing the tool execution.

Format:
Thought:
<A single line of reasoning>
Action:
{"name": "tool_name", "arguments": {"arg1": "val1"}}
"""
        final_sys = sys_prompt + "\n" + instruction + "\nAVAILABLE TOOLS:\n" + json.dumps(tools, indent=2)
        full_history = [{"role": "system", "content": final_sys}] + user_msgs
        resp = client.chat(full_history)
        
        try:
            action_split = resp.split("Action:")[-1].strip()
            start = action_split.find('{')
            end = action_split.rfind('}')
            if start != -1 and end != -1:
                return json.loads(action_split[start:end+1]), resp
        except:
            pass
            
        match = re.search(r'\{[^{}]*\"name\"[^{}]*\}', resp)
        if match:
            try:
                return json.loads(match.group(0)), resp
            except:
                pass
        
        # --- SIERRA HACK: Fallback to 'respond' if no JSON is found ---
        return {"name": "respond", "arguments": {"content": resp}}, resp
    else:
        # Default to strict JSON attempt
        resp = client.chat([{"role": "system", "content": sys_prompt}] + user_msgs)
        try:
            json_match = re.search(r'\{.*\}', resp, re.DOTALL)
            if json_match:
                return json.loads(json_match.group()), resp
        except:
            pass
        return None, resp

def planner_node(state: PevState) -> Dict:
    """PORTED: Phase 3 Hierarchical Planner."""
    PEVLogger.node("Planner", "Unified Macro-Planning (Phase 3 Logic)...")
    client = ModelClient(mode="agent")
    
    # 1. Failure Analysis
    failure_history = "None."
    if state.failure_log:
        lines = [f"Failure {i+1}: Tried `{f.get('action')}` → Error: {f.get('error', '?')}" for i, f in enumerate(state.failure_log[-6:])]
        failure_history = "\n".join(lines)
    
    # 2. Reflection Check
    reflection_section = f"\nERROR REFLECTION: {state.error_reflection}\n" if state.error_reflection else ""
    
    # 3. Wisdom Check
    wisdom_section = "\nGLOBAL EXPERTISE:\n" + "\n".join([f"- {w}" for w in state.global_wisdom[-10:]]) if state.global_wisdom else ""

    # 4. Rejection Check (Inner Loop Feedback)
    rejection_section = f"\n### PREVIOUS ATTEMPT REJECTED:\n{state.rejection_feedback}\n" if state.rejection_feedback else ""

    sys_prompt = f"""You are the HIERARCHICAL PLANNER. 
Your ONLY tool is 'submit_plan'. Use it to set the objective for the Executor.

12. CURRENT SYSTEM TIME: {state.current_time}
13. STANDARD OPERATING PROCEDURES (SOP):
    - GOAL-LED REASONING: Close the 'GAP' between the user's request and the environment state.
    - TERMINATION RULE: If a 'Reservation ID' is in your MILESTONE DASHBOARD, the task is FINISHED. Your next step MUST be to respond to the user and terminate.
    - SILENT RECEPTIONIST: Resolve identifiers (user_id, email, etc.) ONLY if they contain real values. DO NOT guess placeholders.
    - GROUND-TRUTH PIVOT: If the user mentions a flight NOT in memory, you MUST call 'search_*' FIRST. Never book a flight found only in the conversation.
    - MATH RECONCILIATION: If booking fails with 'amount mismatch', use 'calculate' to find (Price - Certificates) and resubmit with the EXACT resulting number.
    - NO PLACEHOLDERS: Never guess IDs or use fake values. Ask the user if resolution fails.

### CAPABILITIES CATALOG (Scan these for semantic alternatives):
{get_compact_tool_catalog(state.tools_info)}

MEMORY KERNEL:
{format_memory(state.memory)}

FAILURE HISTORY:
{failure_history}

### ELITE PLANNING MANDATE:
- CRITICAL: Prioritize the 'MILESTONE DASHBOARD' over user requests for repeat actions.
- REJECTION AWARENESS: If you see a 'PREVIOUS ATTEMPT REJECTED' block below, you MUST pick a different tool or different arguments. Do NOT stubborn-draft.

{rejection_section}

In your 'Thought:' block, ALWAYS include:
1. KNOWN VARIABLES: (e.g. reservation_id, user_id found)
2. MISSING VARIABLES: (e.g. what info is still needed)
3. REDUNDANCY CHECK: (Look at the Rejection/Memory section and confirm you are NOT repeating a successful tool.)
4. STRATEGY: (Describe the NEXT progress-advancing tool choice.)
"""
    user_msgs = []
    if state.user_conversation:
        sys_prompt += f"\nTASK CONTEXT:\n{state.user_conversation[0]['content']}\n"
    for turn in state.user_conversation[-8:]:
        if turn['role'] != 'system':
            user_msgs.append({"role": turn['role'], "content": turn['content']})

    tools = [{
        "name": "submit_plan",
        "description": "Submit a 1-2 sentence plan for the executor",
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {"type": "string"},
                "task_completed": {"type": "boolean"}
            },
            "required": ["plan", "task_completed"]
        }
    }]
    
    parsed, raw = invoke_with_paradigm(client, sys_prompt, user_msgs, tools, "react")
    
    if parsed:
        plan = parsed.get("arguments", {}).get("plan") or parsed.get("plan") or "Carry out next step."
        completed = parsed.get("arguments", {}).get("task_completed") or parsed.get("task_completed") or False
        return {"current_plan": plan, "task_completed": completed, "node_logs": [{"node": "planner", "log": raw}]}
    
    return {"current_plan": "Proceed with task.", "node_logs": [{"node": "planner", "error": "Failed parse", "log": raw}]}

def executor_node(state: PevState) -> Dict:
    """PORTED: Phase 3 Executor."""
    PEVLogger.node("Executor", "Mapping plan to tool schema...")
    client = ModelClient(mode="agent")

    # Rejection Check (Inner Loop Feedback)
    rejection_section = f"\n### PREVIOUS ATTEMPT REJECTED:\n{state.rejection_feedback}\n" if state.rejection_feedback else ""
    
    sys_prompt = f"""You are the EXECUTOR. Select the EXACT tool call for the PLAN.

PLAN: {state.current_plan}

MEMORY:
{format_memory(state.memory)}

{rejection_section}

RULES:
- Use `respond` for ALL user-facing messages.
- Use `transfer_to_human_agents` only as a last resort.
- NO conversational filler. JSON only.

### ELITE EXECUTION MANDATE:
Before outputting the 'Action:', confirm that every argument value is present in the MEMORY. If it is not, call a search tool instead.
"""
    # Build tool list including virtual tools
    tools = state.tools_info.copy()
    tools.append({"name": "respond", "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}})
    tools.append({"name": "transfer_to_human_agents", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}}})

    user_msgs = [{"role": m['role'], "content": m['content']} for m in state.user_conversation[-6:] if m['role'] != 'system']
    
    parsed, raw = invoke_with_paradigm(client, sys_prompt, user_msgs, tools, "react")
    
    if isinstance(parsed, str):
        parsed = {"name": parsed, "arguments": {}}
        
    return {"drafted_tool_call": parsed, "node_logs": [{"node": "executor", "raw": raw}]}

def validator_node(state: PevState) -> Dict:
    """PORTED: Phase 3 Pre-flight Validator."""
    if not state.drafted_tool_call or state.drafted_tool_call.get("name") == "respond":
        return {"node_logs": [{"node": "validator", "status": "approved (fast-path)"}]}
        
    PEVLogger.node("Validator", "Pre-flight validation...")
    client = ModelClient(mode="agent")
    
    sys_prompt = f"""You are the VALIDATOR. Predict if the ACTION will SUCCEED or FAIL.

ACTION: {json.dumps(state.drafted_tool_call)}
MEMORY (API RESULTS): {format_memory(state.memory)}

REJECT ONLY IF:
1. Arguments are hallucinated (not in memory AND not in conversation).
2. Action repeats a failed attempt with EXACTLY SAME arguments (same tool AND same values). Note: 'user_id' vs 'mia_li_3668' are NOT the same; you MUST approve if the value changes.
3. Preconditions are fundamentally impossible.

FOUNDATIONAL EXCEPTION: Always APPROVE 'get_*' or 'find_*' tools if they are resolving a NEW identifier mentioned by the user.

Output JSON: {{"decision": "APPROVE"|"REJECT", "reason": "..."}}
"""
    # --- ELITE TOOL VALIDATION ---
    # --- HARD PLACEHOLDER BLOCK ---
    if any(p in json.dumps(drafted_args).lower() for p in ["user_id", "not_provided", "your_id"]):
        msg = "REJECT: Placeholder argument detected. You MUST wait for a real ID from the conversation or memory."
        return {"rejection_feedback": msg, "rejection_source": "validator", "node_logs": [{"node": "validator", "rejection": msg}]}

    # --- MATH & PRICE GUARD ---
    if drafted_name == "book_reservation":
        payments = drafted_args.get("payment_methods", [])
        total_paid = sum(p.get("amount", 0) for p in payments)
        cabin = drafted_args.get("cabin", "economy")
        expected_price = 0
        flight_num_list = [f.get("flight_number") for f in drafted_args.get("flights", [])]
        for m in state.memory:
            if m.get("type") == "tool_result" and "prices" in str(m.get("api_observation")):
                try:
                    # Very basic extraction - in production we'd use a better parser
                    obs_data = json.loads(str(m.get("api_observation")).replace("API output: ", ""))
                    if isinstance(obs_data, list):
                        for flight in obs_data:
                            if flight.get("flight_number") in flight_nums:
                                expected_price += flight.get("prices", {}).get(cabin, 0)
                except: pass

        if expected_price > 0 and total_paid != expected_price:
            msg = f"MATH ERROR: Total payment ({total_paid}) does not match expected price ({expected_price}) for {cabin} class. Use 'calculate' to find the correct split."
            return {"rejection_feedback": msg, "rejection_source": "validator", "node_logs": [{"node": "validator", "rejection": msg}]}
        
        if any(p.get("amount", 0) <= 0 for p in payments):
            msg = "MATH ERROR: Payment amounts must be positive numbers."
            return {"rejection_feedback": msg, "rejection_source": "validator", "node_logs": [{"node": "validator", "rejection": msg}]}
    if drafted_name not in valid_names:
        msg = f"INVALID TOOL: '{drafted_name}' is NOT a real tool. Use ONLY from the available tool list. Do NOT try to use virtual tools like 'think' or 'evaluate'."
        return {
            "rejection_feedback": msg,
            "rejection_source": "validator",
            "node_logs": [{"node": "validator", "rejection": msg}]
        }

    # --- SMART LOOP DETECTION ---
    for m in state.memory:
        prev_obs = str(m.get("api_observation", "")).lower()
        is_prev_success = not any(kw in prev_obs for kw in ["error", "invalid", "fail", "not found", "mismatch"])
        
        if m.get("action_taken") == drafted_name and m.get("arguments_used") == drafted_args:
            if is_prev_success:
                msg = f"[V5.1] REDUNDANCY: Action '{drafted_name}' was already SUCCESSFUL. You already have this result in the MILESTONE DASHBOARD. Move to a DIFFERENT tool."
                return {"rejection_feedback": msg, "rejection_source": "validator", "node_logs": [{"node": "validator", "rejection": msg}]}
            else:
                PEVLogger.info(f"Validator: Allowing retry of previously failed action '{drafted_name}'...")

    parsed, raw = invoke_with_paradigm(client, sys_prompt, [], [], "json")
    if parsed and parsed.get("decision") == "REJECT":
        return {
            "rejection_feedback": parsed.get("reason"),
            "rejection_source": "validator",
            "node_logs": [{"node": "validator", "rejection": parsed.get("reason")}]
        }
    return {"rejection_feedback": None, "node_logs": [{"node": "validator", "status": "approved"}]}

def error_reflection_node(state: PevState) -> Dict:
    """PORTED: Phase 3 Metacognitive failure analyst."""
    PEVLogger.node("Reflection", "Diagnosing failure root-cause...")
    client = ModelClient(mode="agent")
    
    recent_errors = [m for m in state.memory if m.get('type') == 'tool_error'][-3:]
    if not recent_errors:
        return {"error_reflection": None}
        
    error_summary = "\n".join([f"Action: {e.get('action_taken')} | Error: {e.get('api_observation', '?')}" for e in recent_errors])
    
    sys_prompt = f"""You are a METACOGNITION ANALYST. Analyze this failure:
RECENT ERRORS:
{error_summary}

TASK:
1. IDENTIFY ROOT CAUSE.
2. PROPOSE CORRECTIVE PLAN (1-2 sentences).
Output format:
CAUSE: ...
PLAN: ...
"""
    resp = client.chat([{"role": "system", "content": sys_prompt}])
    
    failure_entries = [{
        "action": e.get('action_taken'),
        "args": e.get('arguments_used'),
        "error": e.get('api_observation'),
        "reflection": resp
    } for e in recent_errors]
    
    return {
        "error_reflection": resp,
        "failure_log": state.failure_log + failure_entries,
        "consecutive_error_count": 0,
        "node_logs": [{"node": "reflection", "diagnosis": resp}]
    }

def global_reflector_node(state: PevState) -> Dict:
    """PORTED: Phase 3 Global Expertise Synthesizer."""
    PEVLogger.node("Global Reflector", "Synthesizing persistent insight...")
    client = ModelClient(mode="summarizer")
    
    history_str = str(state.user_conversation[-10:])
    memory_str = str(state.memory[-10:])
    
    sys_prompt = f"""Analyze the failure and output EXACTLY ONE domain-agnostic technical rule for the future.
Example: 'Verify record status before modification'.
HISTORY: {history_str}
MEMORY: {memory_str}
"""
    insight = client.chat([{"role": "system", "content": sys_prompt}])
    return {"global_wisdom": state.global_wisdom + [insight], "node_logs": [{"node": "global_reflector", "insight": insight}]}

def proactive_prefetch(env, state: PevState):
    """PORTED: Phase 3 'Customer Service Heuristic'."""
    obs = state.user_conversation[-1]["content"] if state.user_conversation else ""
    
    # regex matches standard user/reservation ID patterns (e.g. name_name_123 or 6-char alphanumeric)
    user_id_match = re.search(r'\b([a-z]+_[a-z]+_\d{3,6})\b', obs, re.IGNORECASE)
    res_id_match = re.search(r'\b([A-Z\d]{6})\b', obs)
    
    from tau_bench.types import Action
    
    if user_id_match:
        detected_id = user_id_match.group(1).lower()
        tool = next((t for t in state.tools_info if 'user' in t.get('name', '').lower() and 'detail' in t.get('name', '').lower()), None)
        if tool:
            try:
                res = env.step(Action(name=tool['name'], kwargs={"user_id": detected_id}))
                state.memory.append({"action": "AUTO_PREFETCH", "args": {"user_id": detected_id}, "observation": str(res.observation)})
                PEVLogger.success(f"Proactive: Pre-fetched user details for {detected_id}")
            except: pass

    if res_id_match:
        detected_res = res_id_match.group(1)
        tool = next((t for t in state.tools_info if ('reservation' in t.get('name', '').lower() or 'order' in t.get('name', '').lower()) and 'detail' in t.get('name', '').lower()), None)
        if tool:
            try:
                res = env.step(Action(name=tool['name'], kwargs={"reservation_id": detected_res}))
                state.memory.append({"action": "AUTO_PREFETCH", "args": {"reservation_id": detected_res}, "observation": str(res.observation)})
                PEVLogger.success(f"Proactive: Pre-fetched reservation {detected_res}")
            except: pass
