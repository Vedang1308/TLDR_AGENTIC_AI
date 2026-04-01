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
        pass # History is now stored in the state for per-trial isolation

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        PEVLogger.node("Monitor", "Verifying loop safety...")
        
        action = state.current_action_draft
        if not action or "name" not in action:
            return {"is_loop": True, "audit_feedback": "Missing action name."}
        
        # 1. Action Fingerprinting (Domain-Agnostic)
        # We hash the tool name and critical parameters to identify stagnation
        fingerprint = hashlib.md5(json.dumps(action, sort_keys=True).encode()).hexdigest()
        
        # 2. Loop Detection (Stagnation Check) - Check state history
        if fingerprint in state.action_fingerprints:
            PEVLogger.warn(f"Stagnation detected! Fingerprint: {fingerprint}")
            
            # Context-Aware Loop Feedback
            loop_msg = f"Stagnation detected: Already attempted {action['name']} with these arguments."
            if "search" in action["name"].lower():
                loop_msg += " CRITICAL: Result is already in your history. DO NOT repeat the search. Extract and use the IDs and Prices now."
            
            return {
                "is_loop": True, 
                "audit_feedback": loop_msg
            }
            
        # 3. Domain-Agnostic Empty Observation Loop
        tool_obs = [msg["content"] for msg in state.history if msg.get("role") == "tool"]
        if len(tool_obs) >= 3 and all(len(str(obs).strip()) == 0 for obs in tool_obs[-3:]):
            PEVLogger.warn("CRITICAL: Empty Observation Loop stalling execution!")
            return {
                "is_loop": True,
                "audit_feedback": "CRITICAL ALERT: You have performed 3 consecutive thinking/non-informative actions. You are now MANDATED to use the 'respond' tool to provide the user with the options already in your history, or ask for a choice. DO NOT attempt to 'filter' anymore internally."
            }
        elif len(tool_obs) >= 2 and all(len(str(obs).strip()) == 0 for obs in tool_obs[-2:]):
            PEVLogger.warn("Empty Observation Loop detected!")
            return {
                "is_loop": True,
                "audit_feedback": "Stagnation detected: Your last 2 actions yielded NO new information. You are stuck in a non-informative loop. Use a tool that extracts new data or responds to the user."
            }
        
        # Record this fingerprint for future turns in current trial
        state.action_fingerprints.append(fingerprint)
        PEVLogger.success("Action fingerprint is unique.")
        
        return {
            "is_loop": False,
            "node_logs": [{"node": "Monitor", "content": "Action fingerprint verified unique."}]
        }
