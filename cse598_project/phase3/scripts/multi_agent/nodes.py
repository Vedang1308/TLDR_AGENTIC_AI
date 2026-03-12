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

def load_live_wisdom(state: PevState):
    """Refreshes state.global_wisdom from disk for real-time parallel learning."""
    wisdom_file = "results/phase3/persistent_wisdom.json"
    if os.path.exists(wisdom_file):
        try:
            with open(wisdom_file, "r") as f:
                live_wisdom = json.load(f)
                # Merge and deduplicate
                # Using dict.fromkeys to preserve order and deduplicate
                state.global_wisdom = list(dict.fromkeys(state.global_wisdom + live_wisdom))
        except:
            pass

def planner_node(state: PevState) -> Dict:
    llm = get_llm()
    reasoning_mode = os.environ.get("AGENT_REASONING_MODE", "fc")
    
    # Build failure history section — lets the Planner see what already didn't work
    failure_history = ""
    if state.failure_log:
        lines = []
        for i, f in enumerate(state.failure_log[-6:]):  # Last 6 failures max
            lines.append(f"Failure {i+1}: Tried `{f.get('action')}` → Error: {f.get('error', '?')}")
            if f.get('reflection'):
                lines.append(f"  Diagnosis: {f.get('reflection')}")
        failure_history = "\n".join(lines)
    else:
        failure_history = "None."
    
    # LIVE WISDOM RELOAD (Enables real-time learning across parallel trials)
    load_live_wisdom(state)
    all_wisdom = state.global_wisdom
    
    wisdom_section = ""
    if all_wisdom:
        wisdom_section = "\nGLOBAL EXPERTISE (Technical rules learned from past failures — MANDATORY COMPLIANCE):\n"
        # Prioritize newest wisdom
        for i, w in enumerate(all_wisdom[-15:]): 
            wisdom_section += f"- {w}\n"

    # Include Technical Tool Wiki (Prevents hallucinated tool calls)
    tool_wiki_section = ""
    if state.tools_wiki:
        tool_wiki_section = f"\nTECHNICAL TOOL WIKI (Tools available for the Executor — Use these in your plans):\n{state.tools_wiki}\n"

    # Include local error reflection if generated by the reflection node
    reflection_section = ""
    if state.error_reflection:
        reflection_section = f"\n\nERROR REFLECTION (Diagnosis from recent failures in THIS session — CRITICAL):\n{state.error_reflection}\n"

    sys_prompt = """You are the HIERARCHICAL PLANNER for a customer service agent.
Your job is to read the user conversation, memory kernel, and failure history, then use the 'submit_plan' tool to set the next objective.

CRITICAL RULE 1: Do NOT transfer to a human agent unless you have truly exhausted all available tools and approaches.
CRITICAL RULE 2: CHECK THE MEMORY KERNEL before asking the user for any information. If a data retrieval or lookup tool has already executed, the information is already available — use it directly.
CRITICAL RULE 3: Do NOT re-call a data lookup tool whose result is already in the MEMORY KERNEL. Use the existing record to formulate your next action.
CRITICAL RULE 4: Before planning, read the FAILURE HISTORY and GLOBAL EXPERTISE carefully. If previous technical sequences or tool combinations failed, you MUST propose a fundamentally different trajectory.
CRITICAL RULE 5: If the FAILURE HISTORY shows the same tool being rejected multiple times, shift to an alternate tool or a different argument structure.

{wisdom_section}

{tool_wiki_section}

### MANDATORY UNIFORM ACTION RULE:
You are a STRATEGIST, not an executor. You ONLY have access to the `submit_plan` tool.
DO NOT ATTEMPT TO CALL ANY OTHER TOOLS. Use the 'TECHNICAL TOOL WIKI' above to understand what the Executor can do, but your ONLY output must be a `submit_plan` call.

MEMORY KERNEL:
{memory_str}

FAILURE HISTORY (strategies that have already been tried and failed in THIS session):
{failure_history}
{reflection_section}
REJECTION FEEDBACK:
{feedback}
"""
    mem_str = format_memory(state.memory)
    feed_str = f"Source: {state.rejection_source} | Message: {state.rejection_feedback}" if state.rejection_feedback else "None."
    sys_prompt = sys_prompt.format(
        wisdom_section=wisdom_section,
        tool_wiki_section=tool_wiki_section,
        memory_str=mem_str,
        failure_history=failure_history,
        reflection_section=reflection_section,
        feedback=feed_str
    )

    user_msgs = []
    if state.user_conversation:
        user_msgs.append(SystemMessage(content=state.user_conversation[0]['content']))
    for turn in state.user_conversation[-6:]:
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
    
    if parsed_json:
        # ROBUST WRAPPING: If the Planner tried to call a domain tool directly, wrap it as a plan
        name = parsed_json.get("name")
        args = parsed_json.get("arguments", {})
        
        if name == "submit_plan":
            plan = args.get("plan", "Proceed with next steps.")
            if args.get("task_completed", False):
                return {"task_completed": True, "error_reflection": None, "node_logs": [{"node": "planner", "plan": plan, "log": raw_log}]}
        else:
            # Planner hallucinated a domain tool call. Convert it to a plan for the executor.
            plan = f"I will use the {name} tool with these parameters: {json.dumps(args)}. This is to move the task forward."
            print(f"--- [PLANNER RECOVERY] Wrapped direct tool call '{name}' into a plan ---")

        return {
            "current_plan": plan, 
            "rejection_feedback": None,
            "rejection_source": None,
            "error_reflection": None,
            "node_logs": [{"node": "planner", "plan": plan, "log": raw_log}]
        }
    else:
        # Fallback if no JSON at all
        return {
            "current_plan": "Proceed with default action or ask for clarification.",
            "rejection_feedback": None,
            "rejection_source": None,
            "node_logs": [{"node": "planner", "error": "Failed to output plan", "log": raw_log}]
        }

def error_reflection_node(state: PevState) -> Dict:
    """
    METACOGNITION NODE — The 'step back and think' node.
    
    Triggered by multi_agent_strategy.py when the environment returns an API error.
    This node uses the LLM to reason holistically about WHY the error occurred,
    what assumptions were wrong, and what a fundamentally different approach would be.
    
    This is domain-agnostic because it does NOT check hardcoded rules — it asks the LLM
    to use its own knowledge of the environment (tools available, prior conversation context)
    to understand the failure and propose recovery. This mirrors human metacognition:
    'I made a mistake. Why? What should I do differently?'
    """
    llm = get_llm()
    
    # Find the most recent error(s)
    recent_errors = [m for m in state.memory if m.get('type') == 'tool_error'][-3:]
    if not recent_errors:
        return {"node_logs": [{"node": "error_reflection", "status": "no_errors_found"}]}
    
    error_summary = "\n".join([
        f"Action: {e.get('action_taken')} | Args: {json.dumps(e.get('arguments_used', {}))} | Error: {e.get('api_observation', '?')}"
        for e in recent_errors
    ])
    
    failure_history = format_memory(state.memory)
    
    sys_prompt = f"""You are a METACOGNITION ANALYST for an AI agent that just failed.

The agent recently received one or more error responses from the environment:
{error_summary}

Here is everything the agent has done so far in this conversation:
{failure_history}

Your task:
1. Identify the ROOT CAUSE of the error(s). Was it wrong arguments? Wrong tool? Wrong precondition? Wrong ID? Missing data lookup?
2. Explain what the agent believed vs. what was actually true.
3. Propose a CONCISE corrective strategy (1-3 sentences) for the agent's next attempt that avoids the same mistake.

IMPORTANT: Do NOT hardcode domain-specific rules. Reason purely from what the error message tells you and what the prior memory shows.
Do NOT say 'I don't know'. Always produce a diagnosis and corrective plan.
Output format:
ROOT CAUSE: <one line>
CORRECTIVE PLAN: <1-3 sentences on what to do differently>"""
    
    resp = llm.invoke([SystemMessage(content=sys_prompt)])
    reflection_text = resp.content.strip() if resp.content else "Unable to produce reflection."
    
    # Append to failure_log so Planner can see specific failures with their diagnosis
    failure_entries = [{
        "action": e.get('action_taken'),
        "args": e.get('arguments_used'),
        "error": e.get('api_observation'),
        "reflection": reflection_text
    } for e in recent_errors]
    
    return {
        "error_reflection": reflection_text,
        "failure_log": failure_entries,
        "consecutive_error_count": 0,  # Reset after reflection
        "node_logs": [{"node": "error_reflection", "reflection": reflection_text}]
    }


def executor_node(state: PevState) -> Dict:
    llm = get_llm()
    reasoning_mode = os.environ.get("AGENT_REASONING_MODE", "fc")
    
    # Build context from failure log for executor awareness
    failed_actions_note = ""
    if state.failure_log:
        failed_names = list(set(f.get('action') for f in state.failure_log[-4:] if f.get('action')))
        if failed_names:
            failed_actions_note = f"\n\n[ALREADY FAILED]: The following actions have already failed and should NOT be repeated with the same arguments: {failed_names}\n"
    
    sys_prompt = f"""You are the EXECUTOR.
Your ONLY job is to select the exact tool call based on the PLAN provided.

PLAN TO EXECUTE:
{state.current_plan}

MEMORY CONTEXT (Recent past actions):
{format_memory(state.memory)}
{failed_actions_note}
UNIVERSAL TOOL SELECTION PRINCIPLE:
- Use `respond` whenever the plan requires communicating with the user — asking a question,
  requesting clarification, providing information, or confirming a step. This is the ONLY
  tool for user-facing messages.
- Use `transfer_to_human_agents` ONLY when the user has explicitly requested a human agent,
  OR when every available domain tool has been tried and none can resolve the issue.
  This is a PERMANENT, irreversible escalation — do not use it as a substitute for `respond`.
- Use `think` ONLY for internal scratchpad reasoning or calculations. The `think` tool is
  COMPLETELY INVISIBLE to the user. Do not put questions for the user inside `think`."""
    if state.rejection_feedback and state.rejection_source == "syntax_monitor":
        sys_prompt += f"\n\n[CRITICAL]: YOUR PREVIOUS DRAFT WAS REJECTED. Fix this error:\n{state.rejection_feedback}\n"


    tools = state.tools_info.copy()
    tools.append({
        "type": "function",
        "function": {
            "name": "transfer_to_human_agents",
            "description": "PERMANENT escalation to a human agent. Use ONLY when the user explicitly requests a human, or all domain tools are exhausted. Do NOT use this to ask the user a question — use `respond` instead.",
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
    
    # ── PREMATURE TRANSFER GUARD ───────────────────────────────────────────────
    # Domain-agnostic rule: if the agent has fresh un-acted-upon data in memory
    # (e.g. profile lookup results from proactive seeding) but is trying to
    # immediately transfer to a human agent, the transfer is premature.
    # The agent should use that data before escalating.
    if tool_draft.get("name") == "transfer_to_human_agents":
        successful_results = [m for m in state.memory if m.get('type') == 'tool_result']
        if successful_results and len(state.memory) <= 2:
            return {
                "rejection_feedback": (
                    "You have retrieved account/context data in memory but have not used it to "
                    "resolve the user's request. Read the MEMORY KERNEL carefully and proceed "
                    "with the task before considering a transfer to a human agent."
                ),
                "rejection_source": "syntax_monitor",
                "internal_retry_count": current_retries
            }
    
    return {"node_logs": [{"node": "syntax_monitor", "status": "passed"}], "internal_retry_count": 0}



def validator_node(state: PevState) -> Dict:
    """
    UPGRADED: Domain-agnostic LLM pre-flight simulation.
    
    Instead of hardcoded domain-specific rules (which break for new domains),
    the Validator now asks the LLM to SIMULATE the outcome of the proposed action
    BEFORE sending it to the environment. This is analogous to a human thinking
    'if I do X right now, what will happen?' before committing to an action.
    
    The LLM uses the full memory context, the tool schema, and the drafted call
    to predict whether the action will succeed or fail, and reject it proactively
    if it identifies a likely failure.
    """
    llm = get_llm()
    reasoning_mode = os.environ.get("AGENT_REASONING_MODE", "fc")
    current_retries = state.internal_retry_count + 1
    
    # Fast-path for fallback calls generated by syntax_monitor
    if state.drafted_tool_call and state.drafted_tool_call.get("name") in ["respond", "transfer_to_human_agents"] and current_retries >= 5:
        return {"internal_retry_count": 0, "node_logs": [{"node": "validator", "status": "bypassed_for_fallback"}]}
        
    # LIVE WISDOM RELOAD: Access the global hive-mind even in the Validator
    load_live_wisdom(state)
    
    wisdom_section = ""
    if state.global_wisdom:
        wisdom_section = "\nGLOBAL EXPERTISE (Universal Rules from other trials):\n" + "\n".join([f"- {w}" for w in state.global_wisdom])

    # Find matching tool schema to give to the Validator so it can check arguments
    tool_name = state.drafted_tool_call.get("name", "") if state.drafted_tool_call else ""
    matching_schema = next(
        (t for t in state.tools_info if t.get("function", {}).get("name") == tool_name),
        None
    )
    schema_str = json.dumps(matching_schema, indent=2) if matching_schema else "Schema not found."
    
    sys_prompt = f"""You are the VALIDATOR — a critical reasoning agent that performs PRE-FLIGHT SIMULATION.

Before the agent submits an action to the real environment, you must predict whether it will SUCCEED or FAIL.

DRAFTED ACTION:
{json.dumps(state.drafted_tool_call, indent=2)}

TOOL SCHEMA:
{schema_str}

PRIOR MEMORY (what has happened so far):
{format_memory(state.memory)}

FAILURE HISTORY (approaches that already failed):
{chr(10).join([f"`{f.get('action')}` failed: {f.get('error', '?')}" for f in state.failure_log[-4:]]) if state.failure_log else 'None.'}

Your simulation task:
1. Check: Are all required arguments present and non-null?
2. Check: Do the IDs, names, or values in the arguments actually appear in the PRIOR MEMORY? (If an ID was never returned by a previous API call, it may be hallucinated.)
3. Check: Based on the memory, is the precondition for this action satisfied? (e.g., can you modify something that may already be in a terminal state?)
4. Check: Is this action a repetition of something that already failed with the same arguments?
5. Check: GLOBAL EXPERTISE VIOLATION — Does this action violate any of the global technical rules? If so, REJECT.
6. Check: If this is `transfer_to_human_agents` — is this truly the ONLY option left, or is there still a valid tool approach available?

{wisdom_section}

If any check fails, use `declare_verdict` with REJECT and clearly state which check failed and what the agent should do instead.
If all checks pass, use `declare_verdict` with APPROVE.

Be willing to APPROVE when the action logically follows from the memory. Do not be overly restrictive.
The fast-path APPROVE is appropriate for `respond` actions where the agent is simply asking the user a question."""
    
    # Fast-path: Don't invoke LLM for simple respond actions (saves tokens, avoids false rejections)
    if tool_name == "respond":
        return {
            "node_logs": [{"node": "validator", "status": "approved (fast-path respond)"}],
            "internal_retry_count": 0
        }
    
    tools = [{
        "type": "function",
        "function": {
            "name": "declare_verdict",
            "description": "Approve or reject the drafted tool call based on pre-flight simulation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["APPROVE", "REJECT"]},
                    "reason": {"type": "string", "description": "Explain which check failed and what the correct approach is."}
                },
                "required": ["decision"]
            }
        }
    }]

    parsed_json, raw_log = invoke_with_paradigm(llm, sys_prompt, [], tools, reasoning_mode, "Validator")
    
    if parsed_json and parsed_json.get("name") == "declare_verdict":
        args = parsed_json.get("arguments", {})
        if args.get("decision") == "REJECT":
            return {
                "rejection_feedback": args.get("reason", "Rejected by pre-flight simulation"),
                "rejection_source": "validator",
                "internal_retry_count": current_retries,
                "node_logs": [{"node": "validator", "status": f"rejected: {args.get('reason', '')[:100]}"}]
            }
    
    return {
        "node_logs": [{"node": "validator", "status": "approved"}],
        "internal_retry_count": 0
    }

def global_reflector_node(state: PevState) -> Dict:
    """
    Synthesizes a single, domain-agnostic technical insight from a failed task.
    This insight is stored persistently to help future agents avoid the same mistake.
    """
    llm = get_llm()
    
    # We care about the user's initial goals vs. what the agent actually did
    mem_str = format_memory(state.memory)
    conv_history = "\n".join([f"{m['role']}: {m['content']}" for m in state.user_conversation])
    
    # Extract the sequence of plans and errors from the logs
    trace_steps = []
    for log in state.node_logs:
        node = log.get("node", "unknown")
        content = log.get("plan") or log.get("status") or log.get("error") or ""
        trace_steps.append(f"[{node}]: {content}")
    trace_str = "\n".join(trace_steps[-10:]) # last 10 steps

    # LIVE WISDOM RELOAD for comparison
    load_live_wisdom(state)
    existing_wisdom_str = "\n".join([f"- {w}" for w in state.global_wisdom]) if state.global_wisdom else "None."

    sys_prompt = """You are a META-COGNITIVE ARCHITECT. 
Your task is to analyze a failed agent trajectory and synthesize exactly ONE domain-agnostic technical insight.
This insight will be used by future agents to avoid similar failures.

EXISTING GLOBAL WISDOM (Do NOT repeat or duplicate these):
{existing_wisdom}

FAILURE ANALYSIS DATA:
1. USER CONVERSATION HISTORY:
{conversation}

2. EXECUTION TRACE (Last 10 steps):
{trace}

3. MEMORY KERNEL AT FAILURE:
{memory}

GUIDELINES:
1. Be technical and procedural (e.g., "Verify record state before submission").
2. Be domain-agnostic (no names or specific values).
3. Be UNIQUE. If the failure is already explained by one of the EXISTING GLOBAL WISDOM rules, respond with exactly "REDUNDANT".
4. Focus on the HIGHEST-LEVEL technical mistake.
"""
    prompt = sys_prompt.format(
        existing_wisdom=existing_wisdom_str,
        conversation=conv_history,
        trace=trace_str,
        memory=mem_str
    )
    
    resp = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Synthesize the meta-insight for this failure.")])
    insight = resp.content.strip()
    
    # Clean up any "Insight:" prefixes
    insight = re.sub(r'^(Insight|Meta-Insight):\s*', '', insight, flags=re.IGNORECASE)
    
    if "REDUNDANT" in insight.upper() and len(insight) < 20:
        return {
            "global_wisdom": [], 
            "node_logs": [{"node": "global_reflector", "status": "redundant_insight_skipped"}]
        }

    return {
        "global_wisdom": [insight], # This will be appended to the global state
        "node_logs": [{"node": "global_reflector", "insight": insight}]
    }
