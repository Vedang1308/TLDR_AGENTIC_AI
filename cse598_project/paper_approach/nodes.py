import os
import json
import re
from typing import Dict, Any, List
from .state import PevState

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def get_llm(model_name: str, temperature: float = 0.0):
    """
    Returns a ChatOpenAI (vLLM-compatible) instance for the given model.
    """
    api_base = os.getenv("AGENT_API_BASE", "http://localhost:8000/v1")
    return ChatOpenAI(
        model=model_name,
        openai_api_base=api_base,
        openai_api_key="sk-1234",
        temperature=temperature
    )

def summarizer_node(state: PevState):
    """
    [Context Distiller]
    Compresses long conversation history into a Strategic Kernel.
    """
    print(f"--- [SUMMARIZER] ---")
    history = state.get("messages", [])
    if not history:
        return {"strategic_kernel": "Initial turn. No previous summary."}
    
    llm = get_llm(os.getenv("SUMMARIZER_MODEL", "Qwen/Qwen3-4B-Instruct"))
    
    prompt = f"""Summarize the following conversation history into a 'Strategic Kernel'. 
Focus only on:
1. User's current intent and constraints.
2. Verified facts (IDs, names, statuses).
3. Actions already attempted and their outcomes.

Do not include conversational filler. Keep it under 200 words.

HISTORY:
{json.dumps(history[-10:], indent=2)}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"strategic_kernel": response.content}

def strategist_node(state: PevState):
    """
    [Planner]
    Focuses on high-level logic based on the summary and global wisdom.
    """
    print(f"--- [STRATEGIST] ---")
    llm = get_llm(os.getenv("AGENT_MODEL_NAME", "Qwen/Qwen3-32B"))
    
    wisdom = "\n".join([f"- {w}" for w in state.get("global_wisdom", [])])
    
    system_prompt = f"""You are the Lead Strategist in the PEVAL architecture. 
Your goal is to satisfy the user request using the STRATEGIC KERNEL and GLOBAL WISDOM.

STRATEGY: {os.getenv('AGENT_STRATEGY', 'ReAct')}
{ "FOCUS: Analyze previous outcomes and correct mistakes." if state.get('metadata', {}).get('loop_warning') else "FOCUS: Efficient task execution." }

RULES:
- Output a single high-level technical intent (e.g., 'Verify reservation XYZ then attempt cancellation').
- Adhere strictly to GLOBAL WISDOM.
- If the last action failed (check kernel), explain WHY it failed and propose a DIFFERENT approach.

GLOBAL WISDOM:
{wisdom}

STRATEGIC KERNEL:
{state.get('strategic_kernel', 'None')}
"""
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"USER: {state['user_utterance']}")
    ])
    return {"strategic_plan": response.content}

def tactician_node(state: PevState):
    """
    [Executor]
    Converts strategic instructions into tool-call drafts.
    """
    print(f"--- [TACTICIAN] ---")
    llm = get_llm(os.getenv("AGENT_MODEL_NAME", "Qwen/Qwen3-32B"))
    
    plan = state.get("strategic_plan", "")
    system_prompt = f"""You are the Tactician. Your job is to convert the STRATEGIC PLAN into a specific tool call.

STRATEGIC PLAN: {plan}

Output ONLY the tool call in JSON format.
Example: {{"name": "tool_x", "arguments": {{"id": "123"}} }}
"""
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Intent: {state['user_utterance']}")
    ])
    
    try:
        match = re.search(r'(\{.*\})', response.content, re.DOTALL)
        draft = json.loads(match.group(1)) if match else {}
    except:
        draft = {"name": "respond", "arguments": {"content": "I encountered an error preparing the action."}}
        
    return {"action_draft": draft}

def translator_node(state: PevState):
    """
    [Semantic Translator]
    Normalizes parameters and ensures schema compliance.
    """
    print(f"--- [TRANSLATOR] ---")
    draft = state.get("action_draft", {})
    normalized = draft.copy()
    if "arguments" in normalized:
        for k, v in normalized["arguments"].items():
            if isinstance(v, str):
                normalized["arguments"][k] = v.strip()
    return {"normalized_action": normalized}

def monitor_node(state: PevState):
    """
    [Outcome Monitor]
    A deterministic check for logical loops or safety violations.
    """
    print(f"--- [MONITOR] ---")
    action = state.get("normalized_action", {})
    history = state.get("messages", [])
    
    last_agent_actions = [m for m in history if m.get("role") == "assistant" and "tool_calls" in m]
    if last_agent_actions:
        last_action = last_agent_actions[-1]["tool_calls"][0]["function"]
        if last_action["name"] == action.get("name") and last_action["arguments"] == action.get("arguments"):
            print("--- [LOOP DETECTED] ---")
            return {"current_node": "strategist", "metadata": {"loop_warning": True}}
            
    return {"current_node": "validator"}

def validator_node(state: PevState):
    """
    [Deterministic Guardrail -> Env Dispatch]
    Dispatches the normalized action to the environment and handles observations.
    """
    print(f"--- [VALIDATOR] ---")
    action = state.get("normalized_action", {})
    if not action or "name" not in action:
        return {"env_observation": "No action provided.", "current_node": "strategist"}
    return {"current_node": "env_update"}

def learning_node(state: PevState):
    """
    [Self-Reflection & Insight Synthesis]
    Analyzes the full session history to generate technical rules.
    """
    print(f"--- [LEARNING NODE] ---")
    if not state.get("is_finished") or (state.get("reward", 0) >= 1.0):
        return {}
        
    llm = get_llm(os.getenv("REFLECTOR_MODEL", "Qwen/Qwen3-32B"))
    history = state.get("messages", [])
    
    system_prompt = """You are a Research Analyst. Identify the technical 'root cause' and formulate a ONE-LINE RULE.
RULE: <Technical Rule>
"""
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"TRAJECTORY:\n{json.dumps(history, indent=2)}")
    ])
    
    match = re.search(r'RULE:\s*(.*)', response.content)
    if match:
        new_rule = match.group(1).strip()
        print(f"--- [NEW WISDOM SYNTHESIZED]: {new_rule} ---")
        return {"global_wisdom": state.get("global_wisdom", []) + [new_rule]}
    return {}
