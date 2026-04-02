from typing import Dict, Any
from ..core.state import PEVState
from ..core.model_client import ModelClient

class ContextDistiller:
    """
    Component: Semantic Context Distiller / Summarizer
    Role: Uses OpenAI (No local VRAM cost) to compress history.
    """
    def __init__(self):
        self.client = ModelClient(mode="summarizer")
        self.system_prompt = (
            "You are a Context Distiller. Your goal is to compress history into a dense 'Strategic Kernel'. \n"
            "0. INITIAL FACT-ANCHORING: If the input is Turn 0 (Initial Task), you MUST perform a 'Fact Harvest'. Extract all IDs, names, dates, and preferences provided in the instruction and label them as 'GROUND TRUTH (STATED IN TASK)'. \n"
            "1. WORLD SNAPSHOT: Update the following:\n"
            "- Task-Relevant Entities and Value-Attribute pairs.\n"
            "- All Unique Identifiers (IDs, Codes, References) and associated numerical values (Prices, Costs, Quantities).\n"
            "- Explicitly flag 'Information Provided in Task' versus 'Information to be Discovered'.\n"
            "- Current User Preferences or constraints (e.g., specific times, limits, requested items).\n"
            "Remove all conversational fluff and repeated tool-call logs. The output MUST be domain-agnostic."
        )

    def distill_observation(self, name: str, args: Dict[str, Any], raw_output: str) -> str:
        """Proactively summarizes large tool outputs to prevent context bloat."""
        print(f"--- [NODE] Observation Distiller (Summarizing {name}) ---")
        prompt = [
            {"role": "system", "content": (
                "You are an Observation Architect. Your goal is to convert long, unstructured "
                "tool outputs into a dense summary. \n"
                "FORMAT: 'ToolName(Args) -> [Key Data Only]'. \n"
                "CRITICAL: Keep all Unique Identifiers, numerical values, and actionable data points, but remove formatting fluff and boilerplate."
            )},
            {"role": "user", "content": f"Summarize this Tool Output:\nFunction: {name}\nArgs: {args}\nRaw Output: {raw_output}"}
        ]
        summary = self.client.chat(prompt)
        return f"[OBSERVATION_SUMMARY] {name}({args}) -> {summary}"

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        # History-level distillation (Fallback)
        if len(str(state.history)) < 4000:
            return {"summary": str(state.history)}

        print("--- [NODE] History Distiller (Compressing) ---")
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"History to distill: {state.history}"}
        ]
        summary = self.client.chat(prompt)
        return {"summary": summary}
