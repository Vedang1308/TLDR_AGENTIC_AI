import os
import json
from ..core.state import PEVState
from ..core.config import PEVConfig

class WisdomStore:
    """
    Component: Global Wisdom Store
    Role: Manages persistent cross-task expertise.
    """
    def __init__(self, wisdom_file: str):
        self.wisdom_file = wisdom_file
        os.makedirs(os.path.dirname(self.wisdom_file), exist_ok=True)

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        print("--- [NODE] Wisdom Store (Retrieving) ---")
        
        if not os.path.exists(self.wisdom_file):
            return {"global_wisdom": []}

        try:
            with open(self.wisdom_file, "r") as f:
                expertise = json.load(f)
            
            # Simple Semantic RAG: Filter expertise based on current task context
            # (In production, this would use vector embeddings)
            keywords = ["airline", "retail", "cancel", "refund", "seat"]
            active_keywords = [k for k in keywords if any(k in str(state.history).lower() for k in keywords)]
            
            relevant_wisdom = [ins for ins in expertise if any(k in ins.lower() for k in active_keywords)]
            
            return {
                "global_wisdom": relevant_wisdom[:5], # Don't bloat the state
                "node_logs": [{"node": "WisdomStore", "content": f"Retrieved {len(relevant_wisdom)} insights."}]
            }
        except:
            return {"global_wisdom": []}

    def save_insight(self, insight: str):
        """Atomic save to prevent corruption."""
        if not insight: return
        
        current = []
        if os.path.exists(self.wisdom_file):
            try:
                with open(self.wisdom_file, "r") as f:
                    current = json.load(f)
            except: pass

        updated = list(dict.fromkeys(current + [insight]))
        temp_file = self.wisdom_file + ".tmp"
        try:
            with open(temp_file, "w") as f:
                json.dump(updated, f, indent=2)
            os.replace(temp_file, self.wisdom_file)
        except:
            if os.path.exists(temp_file): os.remove(temp_file)
