from typing import Dict, Any, List
from ..core.state import PEVState
from ..core.model_client import ModelClient
from ..core.logger import PEVLogger

class Tactician:
    """
    Component: Task Executor
    Role: Translates strategic instructions into technical tool drafts.
    """
    def __init__(self, tools_info: list):
        self.client = ModelClient(mode="agent")
        self.tools_info = tools_info
        
        from ..core.config import PEVConfig
        self.strategy = PEVConfig.TOOL_STRATEGY

        if self.strategy == "react":
            self.system_prompt = (
                "You are the Deterministic Tactician in a PME network.\n"
                "Your ONLY role is to map the 'refined_tactical_plan' to actual tool schema.\n\n"
                "RULES:\n"
                "1. DO NOT reason. Extract parameter values from the 'world_snapshot' (Entities/Constraints).\n"
                "2. If the roadmap step asks to retrieve information from the user (e.g. 'Ask user for ID'), you MUST draft a 'respond' action to ask for it.\n"
                "3. Output STRICTLY as a JSON object with 'action'.\n\n"
                "OUTPUT FORMAT:\n"
                "{\"action\": {\"name\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}}\n\n"
                f"Tools available: {self.tools_info}"
            )
        else:
            self.system_prompt = (
                "You are the PEVAL Tactician. Map the instruction to a tool. "
                "Output your response STRICTLY as a JSON object with 'name' and 'arguments'.\n"
                f"Tools available: {self.tools_info}"
            )

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        PEVLogger.node("Tactician", "Mapping tactical plan to schema...")
        
        feedback = ""
        if state.audit_feedback:
            feedback = f"\n\n[CRITICAL ERROR]: Your previous draft was rejected: {state.audit_feedback}"
            
        blackboard_data = str(state.manifest.model_dump())
        
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Blackboard SSO: {blackboard_data}\n{feedback}"}
        ]
        
        response = self.client.chat(prompt)
        
        # Draft the tool call (Strictly formatted for the Translator to follow)
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            parsed_json = json.loads(json_match.group()) if json_match else {}
            
            if self.strategy == "react" and "action" in parsed_json:
                drafted_call = parsed_json["action"]
                node_log_content = f"Drafted: {drafted_call}"
            else:
                drafted_call = parsed_json
                node_log_content = f"Drafted: {drafted_call}"
            
            PEVLogger.info(f"Action: {drafted_call}")
                
        except:
            drafted_call = {"error": "Failed to parse tactician output", "raw": response}
            node_log_content = f"Failed Parse: {response}"
            PEVLogger.error(f"Failed Parse: {response}")

        return {
            "current_action_draft": drafted_call,
            "node_logs": [{"node": "Tactician", "content": node_log_content}]
        }
