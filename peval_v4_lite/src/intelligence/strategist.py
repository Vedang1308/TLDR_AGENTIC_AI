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
            "1. INGREDIENTS (NER): You MUST extract all Named Entities from the user's initial prompt AND any successful tool-responses (e.g. user_ids, flight_ids, prices). This is your source of truth for tool arguments.\n"
            "2. STATE MACHINE: You MUST track the status of required tools in your JSON (e.g. 'search_direct': 'COMPLETED/NONE', 'search_onestop': 'COMPLETED_ID_HAT069', 'respond': 'PENDING').\n"
            "3. NO REPETITION: If a tool-call has already been performed successfully, DO NOT repeat it. MOVE TO ANALYSIS or RESPONSE.\n"
            "4. NO DEBUGGING: If you see an [INSTRUCTION UPDATE], your previous plan was redundant. Shift strategy to a new and unique path.\n\n"
            "OUTPUT FORMAT: You MUST structure your response exactly as follows:\n"
            "INGREDIENTS (JSON): { \"user_id\": \"...\", \"available_options\": [{\"id\": ..., \"price\": ...}], \"status\": {\"search\": \"COMPLETED\", ...} }\n"
            "PROGRESS: [What has been checked off, or what searches failed so far]\n"
            "OBJECTIVE: [The specific natural language instruction for the tactician to execute next]"
        )

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        PEVLogger.node("Strategist", "Planning next objective...")
        
        from ..core.config import PEVConfig
        
        # HYBRID CONTEXT: Combine global archive (summary) with very recent raw turns (to prevent timeouts)
        summary_part = f"GLOBAL ARCHIVE (Summary): {state.summary}\n" if state.summary else ""
        recent_history = f"LATEST RAW TURNS: {str(state.history[-5:])}"
        context = f"{summary_part}{recent_history}"
        
        # Include feedback from the Auditor/Monitor if the previous attempt was rejected
        feedback = ""
        if state.audit_feedback:
            feedback = f"\n\n[INSTRUCTION UPDATE]: {state.audit_feedback}\nLatest Observation: {state.last_observation}"
            
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context: {context}\nKnowledge: {state.memory_kernel}\nPersistent Ingredients: {state.persistent_ner}{feedback}\nStrategy requested:"}
        ]
        
        objective = self.client.chat(prompt)
        PEVLogger.info(f"Objective: {objective}")
        
        # Extract Persistent NER (JSON) from the Strategist's own output
        import json
        import re
        ingredients = {}
        ner_match = re.search(r'INGREDIENTS \(JSON\): (\{.*\})', objective)
        if ner_match:
            try:
                ingredients = json.loads(ner_match.group(1))
            except:
                pass
        
        # Update State (Strategist only sets the instruction)
        return {
            "strategic_instruction": objective,
            "persistent_ner": ingredients,
            "node_logs": [{"node": "Strategist", "content": objective}]
        }
