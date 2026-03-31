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
            "You are the PEVAL Architect and Metacognition Specialist. The execution trail is finished.\n"
            "Your goal is to extract one 'Distilled Procedural Rule' to prevent future failures.\n"
            "RULES:\n"
            "1. ANONYMIZE EVERYTHING: Strictly remove names, dates, and specific IDs. Use 'the user', 'the flight', 'the ID'.\n"
            "2. FOCUS ON PROTOCOL: Record *how* to find data or *why* a policy was violated, not the specific data itself.\n"
            "3. BE ACTIONABLE: Start with 'When...', 'Always...', or 'Ensure...'.\n"
            "4. NO THINKING DUMP: Output ONLY the distilled rule string. Do not output tags like '<think>'.\n\n"
            "Format Example: [Policy Alignment] Always confirm user details before database updates."
        )

    def __call__(self, state: PEVState) -> str:
        print("--- [NODE] Global Learner (Synthesizing) ---")
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Final Trajectory: {state.memory_kernel}\nResult: Reward {state.reward}"}
        ]
        
        insight = self.client.chat(prompt)
        return insight
