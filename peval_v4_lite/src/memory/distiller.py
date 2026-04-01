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
            "You are a Context Distiller. Your goal is to compress a long conversation history "
            "into a dense 'Strategic Kernel'. \n"
            "CRITICAL: You MUST keep the following information intact:\n"
            "- All Origin and Destination Cities/Airports.\n"
            "- All Flight Numbers, Dates, and Prices (format these as a DENSE LIST: [ID, Price]).\n"
            "- All Reservation IDs and User IDs.\n"
            "- Current User Preferences (e.g. 'after 11 AM').\n"
            "Remove all conversational fluff and repeated tool-call logs."
        )

    def distill_observation(self, name: str, args: Dict[str, Any], raw_output: str) -> str:
        """Proactively summarizes large tool outputs to prevent context bloat."""
        print(f"--- [NODE] Observation Distiller (Summarizing {name}) ---")
        prompt = [
            {"role": "system", "content": (
                "You are an Observation Architect. Your goal is to convert long, unstructured "
                "tool outputs into a dense summary. \n"
                "FORMAT: 'ToolName(Args) -> [Key Data Only]'. \n"
                "CRITICAL: Keep all IDs, Prices, and Dates, but remove formatting fluff."
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
