from ..core.state import PEVState
from ..core.model_client import ModelClient

class GlobalLearner:
    """
    Component: Global Learning Node
    Role: Cross-task knowledge synthesis (The 'Slow Path').
    """
    def __init__(self):
        self.client = ModelClient(mode="agent")
        self.system_prompt = (
            "You are the PEVAL Global Learner. The task is done. "
            "If the reward is less than 1.0, identify the EXACT REASON for the failure or loop. "
            "Synthesize a 'Technical Rule' into wisdom.json to prevent this specific mistake. "
            "If the task was successful, extract one efficiency tip. "
            "Output ONLY the high-level technical insight string."
        )

    def __call__(self, state: PEVState) -> str:
        print("--- [NODE] Global Learner (Synthesizing) ---")
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Final Trajectory: {state.memory_kernel}\nResult: Reward {state.reward}"}
        ]
        
        insight = self.client.chat(prompt)
        return insight
