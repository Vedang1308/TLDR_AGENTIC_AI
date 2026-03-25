from typing import Dict, Any
from ..core.state import PEVState

class PolicyManager:
    """
    Component: System Specs
    Role: Manages domain-specific policies and constraints.
    """
    def __init__(self, wiki: str):
        self.wiki = wiki

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        """Injects rules into the Memory Kernel."""
        print("--- [NODE] Policy Manager (Syncing) ---")
        
        # We parse the wiki for 'Critical Policies' (Simplified for demo)
        policies = []
        if "refund" in self.wiki.lower():
            policies.append("Refund Policy: Only allow if within 24 hours.")
        if "seat" in self.wiki.lower():
            policies.append("Seat Policy: Business class requires voucher or payment.")
            
        return {
            "node_logs": [{"node": "Policies", "content": f"Injected {len(policies)} constraints."}]
        }
