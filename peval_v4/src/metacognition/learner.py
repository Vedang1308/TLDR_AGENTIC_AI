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
            "You are the PEVAL Global Learner. The task has CONCLUDED. "
            "Analyze the entire memory kernel (successes and failures) and "
            "extract ONE high-level technical insight to help future agents "
            "avoid mistakes in this domain. Output ONLY the insight string."
        )

    def __call__(self, state: PEVState) -> str:
        print("--- [NODE] Global Learner (Synthesizing) ---")
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Final Trajectory: {state.memory_kernel}\nResult: Reward {state.reward}"}
        ]
        
        insight = self.client.chat(prompt)
        return insight
