from typing import Dict, Any
from ..core.state import PEVState
from ..core.model_client import ModelClient
from ..core.logger import PEVLogger

class Auditor:
    """
    Component: Validator Critic
    Role: Zero-Trust audit of logic and policy compliance.
    """
    def __init__(self, domain_policies: str):
        self.client = ModelClient(mode="agent")
        self.domain_policies = domain_policies
        self.system_prompt = (
            "You are the PEVAL Auditor. The Strategist has already planned the logic. "
            "Check the proposed action against domain policies for safety violations ONLY.\n"
            f"Policies: {self.domain_policies}\n"
            "Be extremely concise. Do NOT repeat the agent's reasoning. "
            "Output 'APPROVED' or 'REJECTION: [brief reason]'."
        )

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        PEVLogger.node("Auditor", "Zero-Trust policy check...")
        
        action = state.current_action_draft
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"History: {state.history[-3:]}\nProposed Action: {action}\nIs this safe?"}
        ]
        
        audit_result = self.client.chat(prompt)
        
        if "APPROVED" in audit_result.upper():
            PEVLogger.success("Action passes Domain Policy checks.")
            return {
                "policy_violation": None,
                "node_logs": [{"node": "Auditor", "content": "Action APPROVED"}]
            }
        else:
            PEVLogger.error(f"Action failed audit: {audit_result}")
            return {
                "policy_violation": "REJECTED",
                "audit_feedback": audit_result,
                "node_logs": [{"node": "Auditor", "content": f"REJECTED: {audit_result}"}]
            }
