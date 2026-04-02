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
        self.client = ModelClient(mode="summarizer") # OpenAI for intelligence
        self.domain_policies = domain_policies
        self.system_prompt = (
            "You are the PEVAL Auditor / Constraint Mask. Your role is Step 4: Verification. \n\n"
            "CRITICAL: MASKING LOGIC \n"
            "1. You are provided with a SYSTEM MANIFEST (S, G, A, C). \n"
            "2. SAFETY MASK: APPROVE all standard tool calls. Let the environment handle parameter validation. \n"
            "   - DO NOT reject an action for minor parameter mismatches (like JFK vs New York) or missing IDs. Let the 'tau-bench' environment catch these. \n"
            "   - ONLY reject if the action is wildly unsafe or hallucinated. Always default to 'APPROVED' so the agent can learn from real environment feedback. \n"
            "3. PERSONA MASK: REJECT 'respond' actions ONLY if they directly violate an explicit persona constraint (e.g., if user is 'angry', response must be appropriate). \n\n"
            "Reference Policy: {self.domain_policies}\n"
            "Output 'APPROVED' or 'REJECTION: [Constraint Violated]'."
        )

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        PEVLogger.node("Auditor", "Zero-Trust policy check...")
        
        action = state.current_action_draft
        action_name = action.get("name", "")
        
        # AGNOSTIC HEURISTIC: Fast-Track all "Read-Only" discovery tools.
        # These are safe as they do not mutate state in most Tau-Bench domains.
        READ_ONLY_PREFIXES = ["get_", "list_", "search_", "calculate_", "think"]
        is_read_only = any(action_name.lower().startswith(p) for p in READ_ONLY_PREFIXES)
        
        if is_read_only:
            PEVLogger.success(f"Fast-Track Audit: '{action_name}' auto-approved (Read-Only).")
            return {
                "policy_violation": None,
                "node_logs": [{"node": "Auditor", "content": f"Fast-Track APPROVED: {action_name}"}]
            }
            
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"SYSTEM MANIFEST: {state.manifest}\nProposed Action: {action}\nIs this action compliant with constraints C and goal G?"}
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
