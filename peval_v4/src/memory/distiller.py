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
            "into a dense 'Strategic Kernel'. Maintain all critical facts (IDs, dates, preferences) "
            "but remove conversational noise."
        )

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        # Only distill if history is getting long (Architecture-Aware Scaling)
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
