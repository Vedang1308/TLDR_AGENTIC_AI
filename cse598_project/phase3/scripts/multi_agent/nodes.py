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
        max_tokens=4096,  # Massively increased: Qwen3 evaluates list returns (like flights) extremely thoroughly in <think> tags. 900 was cutting it off.
    )

def planner_node(state: PevState) -> Dict:
    """
    Supervisor Node: Analyzes conversation, checks memory, sets next objective.
    Does NOT output JSON tool calls. Outputs natural language plan.
    """
    print(f"      ↳ [Planner] Analyzing memory and setting objective...")
    llm = get_llm()
    
    # 1. Build context
    sys_prompt = """You are the STRATEGIC PLANNER for a tool-using AI agent in a customer service setting.
Your job is to read the user conversation, the environment's BUSINESS RULES, and the memory kernel, then output a strict 1-2 sentence PLAN for the Executor.

== ENVIRONMENT BUSINESS RULES (The Wiki) ==
{wiki}

CRITICAL RULES:
1. READ MEMORY FIRST. Before planning an action, check the MEMORY KERNEL below. If that action already appears in memory, DO NOT plan it again - proceed to the NEXT logical step.
2. Be API-FIRST. If the user provided necessary authentication details (like ID, email, or zip), look them up via tool immediately.
3. DO NOT ask the user for information retrievable via API (profile details, reservation info, flight options).
4. NEVER transfer to a human agent. In this environment, transferring to a human immediately fails the task (0.0 reward). You MUST solve the problem yourself using available tools, no matter how complex.
5. EXPLICIT TERMINATION: NEVER output [TASK COMPLETED] unless the user explicitly ends the conversation (e.g. 'thank you, bye') or you have successfully executed the final required action (e.g. `book_reservation` or `exchange_items`) and the user needs nothing else. If you are stuck or an API fails, DO NOT output [TASK COMPLETED] - instead, `respond` to the user and explain!
6. If an action was rejected (see Rejection Feedback), pivot the plan to try a different approach.
7. ALREADY LOOKED UP?: Scan the MEMORY KERNEL thoroughly! If an API was ALREADY called successfully, DO NOT CALL IT AGAIN. Proceed to the next logical step.

STRATEGY GUIDE: Read the BUSINESS RULES above carefully. Determine the exact sequence of tools required to fulfill the user's request.

{{strategy_specific_instructions}}

LATEST API RESULT (Did your last action succeed or fail? Read this!):
{latest_api}

MEMORY KERNEL (Past actions and data):
{memory_str}

REJECTION FEEDBACK:
{feedback}

CRITICAL FORMATTING INSTRUCTION:
End your internal <think> reasoning and output your final action plan starting exactly with `[PLAN]`.
"""

    # Branch the prompt behavior based on the specific strategy variant
    if state.strategy == "multi-agent-react":
        strategy_instructions = (
            "You MUST use the <think> tag to reason about your plan first, "
            "then output your final plan after closing the </think> tag."
        )
    else:
        # For 'act' and 'fc', we forbid reasoning.
        strategy_instructions = (
            "CRITICAL: DO NOT use <think> tags or write internal reasoning. "
            "You MUST immediately output your final action plan and nothing else. "
            "Think silently."
        )

    mem_str = json.dumps(state.memory[-10:], indent=2) if state.memory else "No prior history."
    feed_str = f"Source: {state.rejection_source} | Message: {state.rejection_feedback}" if state.rejection_feedback else "None."
    
    # Extract the absolute latest API observation to prevent "lost in the middle" blindness
    latest_api = "None yet."
    if state.memory and state.memory[-1].get("type") == "tool_result":
        latest_api = f"Action: {state.memory[-1].get('action_taken')}\nObservation: {state.memory[-1].get('api_observation')}"
    
    sys_msg = SystemMessage(content=sys_prompt.format(
        latest_api=latest_api,
        memory_str=mem_str, 
        feedback=feed_str,
        strategy_specific_instructions=strategy_instructions,
        wiki=state.wiki
    ))
    
    # Format user convo
    # Always include the system wiki (turn 0) which contains business logic!
    user_msgs = []
    if state.user_conversation:
        user_msgs.append(SystemMessage(content=state.user_conversation[0]['content']))
        
    # include only the most recent conversation to prevent context explosion
    for turn in state.user_conversation[-3:]: 
        if turn['role'] != 'system': # don't duplicate the wiki if it's in the last 3
            user_msgs.append(HumanMessage(content=f"{turn['role']}: {turn['content']}"))
        
    try:
        response = llm.invoke([sys_msg] + user_msgs)
        raw_content = response.content.strip()
    except Exception as e:
        print(f"      ↳ [Planner] Model generation error (e.g., max tokens reached): {e}")
        raw_content = "Ask the user how you can help them further based on the current context."
    print(f"      ↳ [Planner RAW Output] {raw_content[:500]}..." if len(raw_content) > 500 else f"      ↳ [Planner RAW Output] {raw_content}")
    
    import re
    plan = raw_content
    # Robust Qwen3 think-tag stripping
    plan = re.sub(r'<think>.*?</think>', '', plan, flags=re.DOTALL)
    plan = re.sub(r'<think>.*', '', plan, flags=re.DOTALL)
    plan = plan.strip()
    
    # Extract only the text after [PLAN]
    if "[PLAN]" in plan:
        plan = plan.split("[PLAN]")[-1].strip()
    else:
        # Truncate circular post-think rambling to first 3 meaningful lines
        lines = [l.strip() for l in plan.split('\n') if l.strip()]
        noise_starters = ('wait,', 'however,', 'but wait,', 'hmm,', 'actually,')
        clean_lines = [l for l in lines if not l.lower().startswith(noise_starters)]
        plan = ' '.join(clean_lines[:3]) if clean_lines else ' '.join(lines[:3])
        plan = plan.strip()
    
    # If plan ended up empty, use a highly explicit fallback instead of a generic one
    if not plan:
        plan = "Ask the user how you can help them further based on the current context."
    
    if "[TASK COMPLETED]" in plan:
        return {"task_completed": True, "node_logs": [{"node": "planner", "plan": plan}]}
        
    return {
        "current_plan": plan, 
        "node_logs": [{"node": "planner", "plan": plan}]
    }

def executor_node(state: PevState) -> Dict:
    """
    Actor Node: Receives the specific plan and drafts the exact JSON tool call.
    """
    print(f"      ↳ [Executor] Drafting JSON tool call from plan...")
    llm = get_llm().bind(response_format={"type": "json_object"})
    
    tool_schemas = json.dumps(state.tools_info, indent=2)
    memory_dump = json.dumps(state.memory[-10:], indent=2) 
    
    sys_prompt = f"""You are the EXECUTOR. Your ONLY job is to output a JSON tool call based on the PLAN below.

== PLAN TO EXECUTE (MANDATORY - follow this EXACTLY) ==
{state.current_plan}

== AVAILABLE TOOL SCHEMAS ==
{tool_schemas}
Additional tools allowed without schema: "respond", "transfer_to_human_agents"

== PREVIOUS REJECTION FEEDBACK ==
{f"The Validator or Syntax Monitor rejected your last attempt with this error: {state.rejection_feedback}" if state.rejection_feedback else "None. This is a fresh plan."}

== CONTEXT MEMORY (Use this to fill in exact parameter values) ==
{memory_dump}

CRITICAL RULES:
1. Output ONLY a raw JSON object: {{"name": "<tool_name>", "arguments": {{...}}}}
2. Use EXACT tool names and EXACT parameter keys from the schemas above.
3. NEVER invent parameter values like IDs or Codes. Only use values explicitly confirmed in the Context Memory or User Instructions.
4. POPULATE ARRAYS: If a parameter is an array, you MUST fully populate it with the detailed objects or strings found in memory matching the schema. DO NOT output empty arrays `[]` if the data exists.
5. Calculate Math carefully: If pricing requires multiplication (e.g. Flight Price * Number of Passengers), do the math before submitting `payment_methods` amounts.
8. For conversational replies: {{"name": "respond", "arguments": {{"content": "<message>"}}}}"""

    try:
        resp = llm.invoke([SystemMessage(content=sys_prompt)])
        content = resp.content.strip()
    except Exception as e:
        print(f"      ↳ [Executor] Model generation error (e.g., max tokens reached): {e}")
        content = '{"name": "respond", "arguments": {"content": "I encountered an error planning my next step. Could you please hold on a moment?"}}'
    
    import re
    # Robust think-tag stripping
    content_no_think = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content_no_think = re.sub(r'<think>.*', '', content_no_think, flags=re.DOTALL).strip()
    
    # Aggressive JSON extraction: find first { and last }
    first_brace = content_no_think.find('{')
    last_brace = content_no_think.rfind('}')
    
    drafted_tool = None
    if first_brace != -1 and last_brace != -1 and last_brace >= first_brace:
        json_str = content_no_think[first_brace:last_brace+1]
        # Clean up any leftover markdown codeblock markers if they snuck inside the braces
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        try:
            drafted_tool = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"      ↳ [Executor] JSON Parse Error: {e} | Raw string: {json_str}")
            # Fallback for completely trashed JSON: force a harmless conversational turn
            drafted_tool = {"name": "respond", "arguments": {"content": "I encountered an error planning my next step. Could you please hold on a moment?"}}
    else:
        print(f"      ↳ [Executor] No JSON braces found in output: {content_no_think}")
        drafted_tool = {"name": "respond", "arguments": {"content": "I encountered an error processing my plan. Could you please hold on a moment?"}}
            
    print(f"         [Executor JSON Output] {json.dumps(drafted_tool)}")
        
    return {
        "drafted_tool_call": drafted_tool, 
        "node_logs": [{"node": "executor", "raw_output": content}],
        "rejection_feedback": None,
        "rejection_source": None
    }

def syntax_monitor_node(state: PevState) -> Dict:
    """
    Code Monitor / ToolGate: Deterministically validates syntax, schema, and repetition.
    Directly addresses: Tool Parameter Errors & Repetitive Stuck Loops (PDF Section 4.1, 4.6)
    """
    print(f"      ↳ [Syntax Monitor] Validating JSON schema structure...")
    import re
    tool_draft = state.drafted_tool_call
    
    if not tool_draft:
        reason = "Executor failed to output valid JSON. Try again."
        print(f"         [Syntax Monitor REJECTED] {reason}")
        return {"rejection_feedback": reason, "rejection_source": "syntax_monitor", "rejection_count": state.rejection_count + 1}
        
    if "name" not in tool_draft or "arguments" not in tool_draft:
        reason = "JSON missing 'name' or 'arguments' keys."
        print(f"         [Syntax Monitor REJECTED] {reason} | Draft: {tool_draft}")
        return {"rejection_feedback": reason, "rejection_source": "syntax_monitor", "rejection_count": state.rejection_count + 1}
    
    tool_name = tool_draft["name"].lower()
    
    # Block pseudo-tool names that Qwen3 hallucinates instead of real API calls
    FAKE_TOOLS = {"think", "thought", "reasoning", "internal_thought", "chain_of_thought"}
    if tool_name in FAKE_TOOLS:
        valid_names = [t.get("name", "") for t in state.tools_info] + ["respond", "transfer_to_human_agents"]
        reason = f"'{tool_draft['name']}' is NOT a valid tool. You MUST pick a tool name from this list: {valid_names}. For conversational messages use 'respond'."
        print(f"         [Syntax Monitor REJECTED] {reason}")
        return {
            "rejection_feedback": reason,
            "rejection_source": "syntax_monitor",
            "rejection_count": state.rejection_count + 1
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
            reason = f"Tool '{tool_draft['name']}' does not exist. Valid tools: {valid_names}"
            print(f"         [Syntax Monitor REJECTED] {reason}")
            return {
                "rejection_feedback": reason,
                "rejection_source": "syntax_monitor",
                "rejection_count": state.rejection_count + 1
            }
        # Check required parameters are present (only if schema has parameters defined)
        params_schema = get_tool_params(matching_schema)
        if params_schema:
            required = params_schema.get("required", [])
            schema_props = params_schema.get("properties", {})
            args = tool_draft.get("arguments", {})
            
            # --- Auto-Repair Common Qwen3 Hallucinations ---
            # 1. Hallucinating 'time' instead of 'departure_time' (seen in search_direct_flight logs)
            if "time" in args and "departure_time" not in args and "departure_time" in schema_props:
                args["departure_time"] = args.pop("time")
            # 2. Hallucinating raw 'date' string instead of copying memory
            if args.get("user_id") == "user_id":
                args["user_id"] = "required_but_missing" # Force a real rejection reason instead of echoing string
                
            missing = [r for r in required if r not in args]
            if missing:
                reason = f"Tool '{tool_draft['name']}' is missing required parameters: {missing}. Full parameter list: {list(schema_props.keys())}. YOU MUST PROVIDE EXACT ARGUMENTS."
                print(f"         [Syntax Monitor REJECTED] {reason}")
                return {
                    "rejection_feedback": reason,
                    "rejection_source": "syntax_monitor",
                    "rejection_count": state.rejection_count + 1
                }
    # (PDF Section 4.6: Repetitive Stuck Loops)
    recent_failures = [m for m in state.memory[-5:] if m.get('type') == 'tool_error']
    for rf in recent_failures:
        if rf.get('tool_call') == tool_draft:
            reason = "You already tried this exact action and it failed. Formulate a NEW strategy."
            print(f"         [Syntax Monitor REJECTED] {reason}")
            return {"rejection_feedback": reason, "rejection_source": "syntax_monitor", "rejection_count": state.rejection_count + 1}

    # Passed all checks
    return {"node_logs": [{"node": "syntax_monitor", "status": "passed"}]}

def validator_node(state: PevState) -> Dict:
    """
    Actor-Critic Validator: Evaluates drafted tool against domain business policies.
    Directly addresses: Business Logic Errors & Missing Preconditions (PDF Sections 4.2, 4.5)
    """
    print(f"      ↳ [Validator] Checking drafted tool against business logic policies...")
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
1. MISSING PRECONDITIONS: Does the tool require prior observations (e.g., searching before booking) that are NOT in memory? → REJECT
   - *Note: Look closely at memory for `user_interaction` blocks where the user said 'yes' before rejecting bookings!*
2. BUSINESS LOGIC ERRORS: Does it violate a domain policy (e.g., refund to wrong payment type, exceed allowed coupons, cancel without insurance)? → REJECT
3. PREMATURE TRANSFER: Is `transfer_to_human_agents` called when there are still API tools that could resolve the issue? → REJECT
4. JSON SCHEMA IS NOT YOUR JOB: DO NOT REJECT based on exact JSON parameter formatting (e.g., complaining that `payment_methods` contains amounts, or `flights` is missing an origin). The Syntax Monitor already approved the formatting!
5. SAFE ACTION: Is the action logically sound, has all required preconditions met, and follows policy? → APPROVE
6. READ/LOOKUP ACTIONS: Actions like `get_user_profile`, `search_direct_flight` etc. are ALWAYS safe to APPROVE.

Respond with ONLY "APPROVE" or "REJECT: <specific reason>".

DRAFTED TOOL:
{json.dumps(state.drafted_tool_call, indent=2)}

MEMORY (recent observations):
{json.dumps(state.memory[-8:], indent=2)}
"""

    resp = llm.invoke([SystemMessage(content=sys_prompt)])
    decision = re.sub(r'<think>.*?</think>', '', resp.content, flags=re.DOTALL).strip()
    
    if decision.startswith("REJECT"):
        print(f"         [Validator REJECTED] {decision}")
        return {"rejection_feedback": decision, "rejection_source": "validator", "rejection_count": state.rejection_count + 1}
        
    # If approved, the tool call is returned to multi_agent_strategy.py to execute via env.step()
    return {"node_logs": [{"node": "validator", "status": "approved"}]}
