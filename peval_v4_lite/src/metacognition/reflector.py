from typing import Dict, Any
from ..core.state import PEVState
from ..core.model_client import ModelClient

class ErrorReflector:
    """
    Component: Error Reflector (Inner-Loop)
    Role: Diagnoses API/Logic failures and injects corrective feedback.
    """
    def __init__(self):
        self.client = ModelClient(mode="agent")
        self.system_prompt = (
            "You are the PEVAL Error Reflector. The previous tool call FAILED. "
            "Analyze the failure log and provide a CORRECTIVE DIAGNOSIS. "
            "Tell the Strategist exactly what went wrong and how to fix the parameters."
        )

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        if state.consecutive_errors == 0:
            return {}

        print("--- [NODE] Error Reflector (Diagnosing) ---")
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Last Observation: {state.last_observation}\nFailed Call: {state.current_action_draft}"}
        ]
        
        diagnosis = self.client.chat(prompt)
        
        return {
            "audit_feedback": f"CORRECTIVE DIAGNOSIS: {diagnosis}",
            "node_logs": [{"node": "Reflector", "content": f"Diagnosis: {diagnosis}"}]
        }
