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
            "1. CONTINUOUS CHECKLIST: You MUST create and maintain a Mental Checklist of the user's initial requirements (e.g. date, time limits, Class, direct/one-stop flexibility). Cross-check current findings against this checklist.\n"
            "2. CAUTIOUS PROGRESSION: If a primary search (e.g., direct flight) yields no results, DO NOT panic or apologize. Treat it as a chance to consult your checklist for explicitly allowed alternatives (e.g., one-stop flights).\n"
            "3. KNOWLEDGE AUDIT: Before planning, check the 'Knowledge Kernel'. If the required info is already present, SKIP the discovery tool call and plan the next phase.\n"
            "4. NO DEBUGGING: If you see an [INSTRUCTION UPDATE], your previous plan was redundant. This is a LOGICAL error, not a system failure. Shift strategy to a new and unique path.\n\n"
            "OUTPUT FORMAT: You MUST structure your response exactly as follows:\n"
            "CHECKLIST: [List the user requirements]\n"
            "PROGRESS: [What has been checked off, or what searches failed so far]\n"
            "OBJECTIVE: [The specific natural language instruction for the tactician to execute next]"
        )

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        PEVLogger.node("Strategist", "Planning next objective...")
        
        from ..core.config import PEVConfig
        
        # HYBRID CONTEXT: Combine global archive (summary) with recent raw turns (short-term memory)
        summary_part = f"GLOBAL ARCHIVE (Summary): {state.summary}\n" if state.summary else ""
        recent_history = f"RECENT RAW HISTORY: {str(state.history[-10:])}"
        context = f"{summary_part}{recent_history}"
        
        # Include feedback from the Auditor/Monitor if the previous attempt was rejected
        feedback = ""
        if state.audit_feedback:
            feedback = f"\n\n[INSTRUCTION UPDATE]: {state.audit_feedback}\nLatest Observation: {state.last_observation}"
            
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
