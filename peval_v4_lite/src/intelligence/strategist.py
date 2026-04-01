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
            "1. INGREDIENTS (NER): You MUST extract all Named Entities from the user's initial prompt AND all successful tool-responses in history (e.g. user_ids, card_numbers, certificates, and extracted flight_ids, dates, and prices from search observations). This is your 'Mantry' of available data.\n"
            "2. DATA SHIFT: If a search tool has already been performed successfully, DO NOT repeat it. Move to analyzing the 'INGREDIENTS' to find the best match (e.g. 'flight HAT069 is cheapest') and prepare the action.\n"
            "3. NO APOLOGIES: If a search yields no results (e.g., zero direct flights), DO NOT apologize. Simply consult your INGREDIENTS for explicitly allowed alternatives (e.g., one-stop flights).\n"
            "4. NO DEBUGGING: If you see an [INSTRUCTION UPDATE], your previous plan was redundant. Shift strategy to a new and unique path.\n\n"
            "OUTPUT FORMAT: You MUST structure your response exactly as follows:\n"
            "INGREDIENTS (JSON): {\"user_id\": \"...\", \"card\": \"...\", \"available_options\": [{\"id\": ..., \"price\": ...}], \"status\": \"...\"}\n"
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
