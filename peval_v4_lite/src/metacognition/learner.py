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
            "Your goal is to extract one 'Distilled Procedural Rule' to prevent future failures or loops.\n"
            "FAILURE ANALYSIS:\n"
            "1. IDENTIFY LOOPS: If the trail shows the same tool called twice for the same data, pinpoint the error.\n"
            "2. BE ACTIONABLE: Start with 'Never repeat [tool] if [condition]...' or 'Always switch to [action] after [result]...'.\n"
            "3. NO THINKING DUMP: Output ONLY the distilled rule string. Do not output tags like '<think>'.\n\n"
            "Format Example: [Strategy Pivot] Never repeat a search if the previous results are already in history."
        )

    def __call__(self, state: PEVState) -> str:
        print("--- [NODE] Global Learner (Synthesizing) ---")
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Final Trajectory: {state.memory_kernel}\nResult: Reward {state.reward}"}
        ]
        
        insight = self.client.chat(prompt)
        return insight
