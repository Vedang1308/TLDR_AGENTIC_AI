from typing import Dict, Any
from ..core.state import PEVState
from ..core.model_client import ModelClient
from ..core.logger import PEVLogger

class Strategist:
    """
    Component: Hierarchical Planner
    Role: Sets high-level objectives in natural language.
    Constraint: Does not perform tool calls directly.
    """
    def __init__(self):
        self.client = ModelClient(mode="agent")
        self.system_prompt = (
            "You are the PEVAL Strategist, a high-level reasoning agent. "
            "Your goal is to set the NEXT logical step for the Tactician.\n\n"
            "CRITICAL RULES:\n"
            "1. KNOWLEDGE AUDIT: Before planning, check the 'Knowledge Kernel'. If the required information (e.g., User ID, Flight Numbers) is already present, SKIP the discovery tool call and plan the next phase (e.g., booking or responding).\n"
            "2. ADAPTATION: If you see a [CRITICAL SYSTEM ALERT], your previous plan failed. You MUST pivot to a DIFFERENT strategy.\n"
            "3. NO CODE: Output ONLY the strategic objective in natural language. Do NOT output tool calls or JSON."
        )

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        PEVLogger.node("Strategist", "Planning next objective...")
        
        from ..core.config import PEVConfig
        
        # Build the prompt using the Distilled Summary if available, otherwise raw history
        if PEVConfig.TOOL_STRATEGY == "irma" and state.reformulated_observation:
            context = f"Reformulated Input: {state.reformulated_observation}"
        elif state.summary:
            context = state.summary
        else:
            context = str(state.history[-5:])
        
        # Include feedback from the Auditor/Monitor if the previous attempt was rejected
        feedback = ""
        if state.audit_feedback:
            feedback = f"\n\n[CRITICAL SYSTEM ALERt]: {state.audit_feedback}\nLatest Observation: {state.last_observation}"
            
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context: {context}\nKnowledge: {state.memory_kernel}{feedback}\nStrategy requested:"}
        ]
        
        objective = self.client.chat(prompt)
        PEVLogger.info(f"Objective: {objective}")
        
        # Update State (Strategist only sets the instruction)
        return {
            "strategic_instruction": objective,
            "node_logs": [{"node": "Strategist", "content": objective}]
        }
