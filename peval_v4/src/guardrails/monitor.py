import json
from typing import Dict, Any
import hashlib
from ..core.state import PEVState
from ..core.logger import PEVLogger

class OutcomeMonitor:
    """
    Component: Syntax & Loop Monitor
    Role: Deterministic guardrail using Hierarchical Action Fingerprinting.
    """
    def __init__(self):
        self.action_history = set()

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        PEVLogger.node("Monitor", "Verifying loop safety...")
        
        action = state.current_action_draft
        if not action or "name" not in action:
            return {"is_loop": True, "audit_feedback": "Missing action name."}
        
        # 1. Action Fingerprinting (Domain-Agnostic)
        # We hash the tool name and critical parameters to identify stagnation
        fingerprint = hashlib.md5(json.dumps(action, sort_keys=True).encode()).hexdigest()
        
        # 2. Loop Detection (Stagnation Check)
        if fingerprint in self.action_history:
            PEVLogger.warn(f"Stagnation detected! Fingerprint: {fingerprint}")
            return {
                "is_loop": True, 
                "audit_feedback": f"Stagnation detected: Already attempted {action['name']} with these arguments."
            }
        
        # Record this fingerprint for future turns
        self.action_history.add(fingerprint)
        PEVLogger.success("Action fingerprint is unique.")
        
        return {
            "is_loop": False,
            "node_logs": [{"node": "Monitor", "content": "Action fingerprint verified unique."}]
        }
