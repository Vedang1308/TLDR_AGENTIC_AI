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
            "Your goal is to interpret the conversation history and existing knowledge "
            "to set the NEXT step for the Tactician. "
            "Output ONLY the strategic objective in natural language. "
            "Do NOT output tool calls or code."
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
        
        # Update State (Clearing the error flags so the next turn starts fresh)
        return {
            "strategic_instruction": objective,
            "policy_violation": None,
            "audit_feedback": "",
            "node_logs": [{"node": "Strategist", "content": objective}]
        }
