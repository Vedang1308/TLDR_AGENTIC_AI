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
        max_tokens=900,  # Increased from 512: planner needs to complete <think> chain before emitting plan
        stop=["Observation:", "OBSERVATION:"]
    )

def planner_node(state: PevState) -> Dict:
    """
    Supervisor Node: Analyzes conversation, checks memory, sets next objective.
    Does NOT output JSON tool calls. Outputs natural language plan.
    """
    llm = get_llm()
    
    # 1. Build context
    sys_prompt = """You are the STRATEGIC PLANNER for a tool-using AI agent in a customer service setting.
Your job is to read the user conversation and the memory kernel, then output a strict 1-2 sentence PLAN for the Executor.

CRITICAL RULES:
1. READ MEMORY FIRST. Before planning an action, check the MEMORY KERNEL below. If that action already appears in memory, DO NOT plan it again - proceed to the NEXT logical step.
2. Be API-FIRST. If the user provided their user_id, DO NOT ask for it - look it up via tool immediately.
3. DO NOT ask the user for information retrievable via API (profile details, reservation info, flight options).
4. NEVER transfer to a human agent unless all API options are exhausted or the user explicitly demands it.
5. If the task is complete or user says goodbye, output exactly: [TASK COMPLETED]
6. If an action was rejected (see Rejection Feedback), pivot the plan to try a different approach.

STRATEGY GUIDE (use ACTUAL tool names as they appear in tools_info):
- User mentioned user_id AND no user lookup in memory yet? → call `get_user_details` with that user_id
- User profile already in memory? → proceed to NEXT step (search flights, search reservations, etc.)
- Need flight options? → call `search_direct_flight` first, then `search_onestop_flight` if none found
- Need reservation info but no ID? → call `get_user_details` to get reservations list from profile
- Need to modify/cancel? → call `get_reservation_details` then the appropriate action API
- User confirmed booking details? → call `book_reservation` with all required params

MEMORY KERNEL (ACTIONS ALREADY DONE - do not repeat these):
{memory_str}

REJECTION FEEDBACK:
{feedback}
"""

    mem_str = json.dumps(state.memory[-10:], indent=2) if state.memory else "No prior history."
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
    
    import re
    plan = response.content.strip()
    # Robust Qwen3 think-tag stripping:
    # 1. Strip properly closed <think>...</think> blocks
    plan = re.sub(r'<think>.*?</think>', '', plan, flags=re.DOTALL)
    # 2. Strip unclosed <think>... blocks (happen when max_tokens cuts off before </think>)
    plan = re.sub(r'<think>.*', '', plan, flags=re.DOTALL)
    plan = plan.strip()
    
    # If plan ended up empty (all think, no plan text), use the last non-empty raw line as fallback
    if not plan:
        raw_lines = [l.strip() for l in response.content.split('\n') if l.strip() and not l.strip().startswith('<')]
        plan = raw_lines[-1] if raw_lines else "Proceed with the next logical API action based on the conversation."
    
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
    
    sys_prompt = f"""You are the EXECUTOR. Your ONLY job is to output a JSON tool call based on the PLAN below.

== PLAN TO EXECUTE (MANDATORY - follow this EXACTLY) ==
{state.current_plan}

Available tools (use ONLY these exact names):
{json.dumps([t.get('name') or t.get('function', {}).get('name') for t in state.tools_info], indent=2)}
Additional: "respond" (for user messages), "transfer_to_human_agents" (only as last resort)

Rules:
- Output ONLY a raw JSON object, nothing else: {{"name": "<tool_name>", "arguments": {{...}}}}
- Use EXACT tool names from the list above
- DO NOT output 'think', 'thought', or any non-tool name
- For conversational replies: {{"name": "respond", "arguments": {{"content": "<message>"}}}}
- Follow the PLAN above exactly - do not substitute a different tool"""

    resp = llm.invoke([SystemMessage(content=sys_prompt)])
    content = resp.content.strip()
    
    import re
    # Robust think-tag stripping (handles both closed and unclosed tags)
    content_no_think = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content_no_think = re.sub(r'<think>.*', '', content_no_think, flags=re.DOTALL).strip()
    
    # Try to find JSON block between first { and last }
    json_match = re.search(r'\{.*\}', content_no_think, re.DOTALL)
    
    drafted_tool = None
    if json_match:
        json_str = json_match.group(0)
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        try:
            drafted_tool = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e} | Raw string: {json_str}")
            
    return {"drafted_tool_call": drafted_tool, "node_logs": [{"node": "executor", "raw_output": content}]}

def syntax_monitor_node(state: PevState) -> Dict:
    """
    Code Monitor / ToolGate: Deterministically validates syntax, schema, and repetition.
    Directly addresses: Tool Parameter Errors & Repetitive Stuck Loops (PDF Section 4.1, 4.6)
    """
    import re
    tool_draft = state.drafted_tool_call
    
    if not tool_draft:
        return {"rejection_feedback": "Executor failed to output valid JSON. Try again.", "rejection_source": "syntax_monitor"}
        
    if "name" not in tool_draft or "arguments" not in tool_draft:
        return {"rejection_feedback": "JSON missing 'name' or 'arguments' keys.", "rejection_source": "syntax_monitor"}
    
    tool_name = tool_draft["name"].lower()
    
    # Block pseudo-tool names that Qwen3 hallucinates instead of real API calls
    FAKE_TOOLS = {"think", "thought", "reasoning", "internal_thought", "chain_of_thought"}
    if tool_name in FAKE_TOOLS:
        valid_names = [t.get("name", "") for t in state.tools_info] + ["respond", "transfer_to_human_agents"]
        return {
            "rejection_feedback": f"'{tool_draft['name']}' is NOT a valid tool. You MUST pick a tool name from this list: {valid_names}. For conversational messages use 'respond'.",
            "rejection_source": "syntax_monitor"
        }
    
    # --- ToolGate Schema Validation ---
    # Find the matching tool schema from tools_info (PDF Section 4.1: validate against API schemas)
    # Handles both flat format {name: ...} and nested OpenAI format {type: function, function: {name: ...}}
    allowed_no_schema = {"respond", "transfer_to_human_agents"}
    if tool_draft["name"] not in allowed_no_schema:
        def get_tool_name(t):
            return t.get("name") or t.get("function", {}).get("name")
        def get_tool_params(t):
            return t.get("parameters") or t.get("function", {}).get("parameters", {})
        
        matching_schema = next((t for t in state.tools_info if get_tool_name(t) == tool_draft["name"]), None)
        if matching_schema is None:
            valid_names = [get_tool_name(t) for t in state.tools_info if get_tool_name(t)] + list(allowed_no_schema)
            return {
                "rejection_feedback": f"Tool '{tool_draft['name']}' does not exist. Valid tools: {valid_names}",
                "rejection_source": "syntax_monitor"
            }
        # Check required parameters are present (only if schema has parameters defined)
        params_schema = get_tool_params(matching_schema)
        if params_schema:
            required = params_schema.get("required", [])
            schema_props = params_schema.get("properties", {})
            args = tool_draft.get("arguments", {})
            missing = [r for r in required if r not in args]
            if missing:
                return {
                    "rejection_feedback": f"Tool '{tool_draft['name']}' is missing required parameters: {missing}. Full parameter list: {list(schema_props.keys())}",
                    "rejection_source": "syntax_monitor"
                }
    
    # Check for repetitive loop: did we try this EXACT tool call recently and fail?
    # (PDF Section 4.6: Repetitive Stuck Loops)
    recent_failures = [m for m in state.memory[-5:] if m.get('type') == 'tool_error']
    for rf in recent_failures:
        if rf.get('tool_call') == tool_draft:
            return {"rejection_feedback": "You already tried this exact action and it failed. Formulate a NEW strategy.", "rejection_source": "syntax_monitor"}

    # Passed all checks
    return {"node_logs": [{"node": "syntax_monitor", "status": "passed"}]}

def validator_node(state: PevState) -> Dict:
    """
    Actor-Critic Validator: Evaluates drafted tool against domain business policies.
    Directly addresses: Business Logic Errors & Missing Preconditions (PDF Sections 4.2, 4.5)
    """
    import re
    
    # Fast-path: auto-approve read-only tools without LLM call (~60% of actions)
    # These are pure lookups with no business logic risk
    READ_ONLY_TOOLS = {
        "get_user_details", "get_user_profile", "search_direct_flight",
        "search_onestop_flight", "get_reservation_details", "list_all_airports",
        "respond", "transfer_to_human_agents",
    }
    tool_name = state.drafted_tool_call.get("name", "") if state.drafted_tool_call else ""
    if tool_name in READ_ONLY_TOOLS or tool_name.startswith("search_") or tool_name.startswith("get_") or tool_name.startswith("list_"):
        return {"node_logs": [{"node": "validator", "status": "approved (fast-path)"}]}
    
    llm = get_llm()
    
    # Inject the domain business policy wiki (from the system turn) for informed validation
    # Truncated to 2000 chars to keep validator prompt lean and fast
    wiki_context = ""
    if state.user_conversation and state.user_conversation[0].get('role') == 'system':
        full_wiki = state.user_conversation[0]['content']
        wiki_context = full_wiki[:2000] + ("..." if len(full_wiki) > 2000 else "")
    
    sys_prompt = f"""You are the VALIDATOR (Actor-Critic). Your role is to CRITIQUE the drafted tool call below.

BUSINESS DOMAIN POLICIES (key rules only):
{wiki_context}

Your review criteria:
1. MISSING PRECONDITIONS: Does the tool require prior observations (e.g., searching before booking, fetching before modifying) that are NOT in memory? → REJECT
2. BUSINESS LOGIC ERRORS: Does it violate a domain policy (e.g., refund to wrong payment type, exceed allowed coupons, cancel without insurance)? → REJECT
3. PREMATURE TRANSFER: Is `transfer_to_human_agents` called when there are still API tools that could resolve the issue? → REJECT
4. SAFE ACTION: Is the action logically sound, has all required preconditions met, and follows policy? → APPROVE
5. READ/LOOKUP ACTIONS: Actions like `get_user_profile`, `search_direct_flight` etc. are ALWAYS safe to APPROVE.

Respond with ONLY "APPROVE" or "REJECT: <specific reason>".

DRAFTED TOOL:
{json.dumps(state.drafted_tool_call, indent=2)}

MEMORY (recent observations):
{json.dumps(state.memory[-8:], indent=2)}
"""

    resp = llm.invoke([SystemMessage(content=sys_prompt)])
    decision = re.sub(r'<think>.*?</think>', '', resp.content, flags=re.DOTALL).strip()
    
    if decision.startswith("REJECT"):
        return {"rejection_feedback": decision, "rejection_source": "validator"}
        
    # If approved, the tool call is returned to multi_agent_strategy.py to execute via env.step()
    return {"node_logs": [{"node": "validator", "status": "approved"}]}
