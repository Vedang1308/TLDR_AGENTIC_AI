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
    def __init__(self, tools_info: list):
        # User Mandate: Use gpt-5.4-pro / gpt-4o for all intelligence
        self.client = ModelClient(mode="summarizer")
        
        # Generation: Function Mapping (Name, Inputs, Outputs/Goal)
        # Handle both flat and OpenAI-wrapped tool definitions (tau-bench compatible)
        self.function_mapping = []
        for t in tools_info:
            target = t.get("function", t) # Support OpenAI-style wrapping
            name = target.get("name", "Unknown")
            desc = target.get("description", "No description")
            params = target.get("parameters", {}).get("required", [])
            self.function_mapping.append(f"- {name}({', '.join(params)}): {desc}")
        
        self.function_mapping = "\n".join(self.function_mapping)

        self.system_prompt = (
            "You are the High-Precision PEVAL Strategist / Manifest Decomposer. \n"
            "Your goal is to convert an instruction into a System of Constraints (P = <S, G, A, C>) using your Tool Map.\n\n"
            "AVAILABLE TOOL MAP (Function Mapping):\n"
            f"{self.function_mapping}\n\n"
            "CRITICAL: FORMAL MANIFEST DECOMPOSITION \n"
            "1. S (State): identify User ID (MANDATORY), birth_date, and current knowledge. DO NOT use generic placeholders like 'unknown'. Extract the exact string from the user's prompt or leave it out. \n"
            "2. G (Goal): Define the final definition of success. \n"
            "3. A (Actions): Which tools from the map above ARE actually needed for this path? \n"
            "4. C (Constraints): Identify EVERY Rule of the Road. \n"
            "   - DOMAIN NOTE: Airport codes (JFK, EWR, LGA) ARE valid for New York. Do not reject them. \n\n"
            "OUTPUT FORMAT (JSON ONLY):\n"
            "{\n"
            "  \"MANIFEST\": {\n"
            "    \"state_s\": {\"user_id\": \"...\", ...},\n"
            "    \"goal_g\": \"Definition of success\",\n"
            "    \"actions_a\": [\"search_direct_flight\", ...],\n"
            "    \"constraints_c\": {\"hard\": [...], \"soft\": [...], \"persona\": \"...\"}\n"
            "  },\n"
            "  \"objective\": \"The specific natural language instruction for the tactician\"\n"
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
        
        # Format the result for the Engine (P = <S, G, A, C>)
        objective = decision.get("objective", "Execute next logical task step.")
        manifest_data = decision.get("MANIFEST", {})
        
        PEVLogger.info(f"Goal G: {manifest_data.get('goal_g', 'Unknown')}")
        PEVLogger.info(f"Hard Constraints: {manifest_data.get('constraints_c', {}).get('hard', 'None')}")
        
        return {
            "strategic_instruction": objective,
            "manifest": manifest_data, # Triggers deep merge in engine.py
            "node_logs": [{"node": "Strategist", "content": f"Manifest Goal: {manifest_data.get('goal_g')}"}]
        }
