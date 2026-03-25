from ..core.state import PEVState
from ..core.model_client import ModelClient

class InputReformulator:
    """
    Component: Input Reformulator (IRMA)
    Role: Transforms raw, noisy environmental observations or user utterances 
    into structured, canonical inputs for the reasoning nodes.
    """
    def __init__(self):
        self.client = ModelClient(mode="agent")
        self.system_prompt = (
            "You are the Input Reformulator. Your task is to take a raw observation or "
            "user utterance and extract the core actionable information into a clean, "
            "structured format. Remove irrelevant noise. Output only the reformatted observation."
        )

    def __call__(self, state: PEVState) -> dict:
        print("--- [NODE] Input Reformulator (IRMA) ---")
        
        # Determine what to reformulate: the latest item in history
        if not state.history:
            return {"reformulated_observation": ""}
            
        last_entry = state.history[-1]
        
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Raw Input: '{last_entry['content']}'"}
        ]
        
        reformulated = self.client.chat(prompt)
        
        return {
            "reformulated_observation": reformulated,
            "node_logs": [{"node": "Reformulator", "content": f"Reformulated to: {reformulated}"}]
        }
