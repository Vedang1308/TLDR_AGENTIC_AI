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
            "You are the High-Precision PEVAL Strategist. Your goal is to guide an agent "
            "through a task by maintaining a persistent 'INGREDIENTS' checklist. \n\n"
            "CRITICAL: COMPARATIVE DEEP-NER \n"
            "1. STEP 1 (EXTRACTION): You MUST extract the `user_id`, `reservation_id`, `origin`, "
            "`destination`, and all user constraints from the first message. \n"
            "2. TURN-BY-TURN AUDIT: In every turn, compare your CURRENT KNOWLEDGE with the `persistent_ner` "
            "dictionary provided. If new info appeared in the history, APPEND it. If old info (like `user_id`) "
            "is missing from your current draft, YOU MUST RESTORE IT. Never lose data. \n"
            "3. STATE MACHINE: Track tool status: 'search': 'COMPLETED/NONE', 'booking': 'PENDING'. \n\n"
            "OUTPUT FORMAT (JSON ONLY):\n"
            "{\n"
            "  \"thought_ner\": \"Comparison: What is in Persistent Ingredients vs what is in recent history? Is user_id present?\",\n"
            "  \"thought_strategy\": \"Step-by-step reasoning for the next objective\",\n"
            "  \"INGREDIENTS\": { \"user_id\": \"...\", \"status\": {...}, ... },\n"
            "  \"OBJECTIVE\": \"Instructions for the tactician\"\n"
            "}"
        )

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        PEVLogger.node("Strategist", "Planning next objective (Comparative NER)...")
        
        from ..core.config import PEVConfig
        
        # HYBRID CONTEXT: Combine global summary with very recent raw history
        summary_part = f"GLOBAL ARCHIVE (Summary): {state.summary}\n" if state.summary else ""
        recent_history = f"LATEST RAW TURNS: {str(state.history[-5:])}"
        context = f"{summary_part}{recent_history}"
        
        # Include feedback from Audit if the previous attempt failed
        feedback = ""
        if state.audit_feedback:
            feedback = f"\n\n[INSTRUCTION UPDATE]: {state.audit_feedback}"
            
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context: {context}\nPersistent Ingredients: {state.persistent_ner}{feedback}\nIdentify missing entities and set the next objective. Output JSON."}
        ]
        
        decision_raw = self.client.chat(prompt)
        
        # Extract JSON from potential model conversational wrapper
        import json, re
        decision = {}
        json_match = re.search(r'\{.*\}', decision_raw, re.DOTALL)
        if json_match:
            try:
                decision = json.loads(json_match.group())
            except:
                pass
        
        # Format the result for the Engine
        objective = decision.get("OBJECTIVE", "Proceed with task.")
        ner_updates = decision.get("INGREDIENTS", {})
        ner_thought = decision.get("thought_ner", "No audit logs provided.")
        
        PEVLogger.info(f"NER Audit: {ner_thought}")
        PEVLogger.info(f"Objective: {objective}")
        
        return {
            "strategic_instruction": objective,
            "persistent_ner": ner_updates, # Triggers deep merge in engine.py
            "node_logs": [{"node": "Strategist", "content": f"NER Comparison: {ner_thought}"}]
        }
