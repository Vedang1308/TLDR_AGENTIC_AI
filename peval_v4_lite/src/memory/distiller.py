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

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        # Lowered threshold to manage Gaudi/72B inference latency
        if len(str(state.history)) < 2000:
            return {"summary": str(state.history)}

        print("--- [NODE] Context Distiller (Compressing) ---")
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"History to distill: {state.history}"}
        ]
        
        summary = self.client.chat(prompt)
        
        return {
            "summary": summary,
            "node_logs": [{"node": "Distiller", "content": "History compressed."}]
        }
