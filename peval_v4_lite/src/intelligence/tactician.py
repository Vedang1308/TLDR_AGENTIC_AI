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
                "You are the Deterministic Tactician. Your goal is to map a strategic intent to a technical tool call.\n\n"
                "AGNOSTIC SCHEMA MAPPING RULES:\n"
                "1. SCHEMA VALIDATION: Scan the 'AVAILABLE TOOLS' (self.tools_info) to find the tool for 'refined_tactical_plan'.\n"
                "2. PARAMETER RESOLUTION (PRE-EXECUTION CHECK): Scan the 'world_snapshot' FIRST to fulfill every REQUIRED parameter. If a value is missing from the instruction but exists in the snapshot (Variable Storage), you MUST inject it automatically.\n"
                "3. MULTI-VALUE HANDLING: If multiple candidates exist for a variable, prioritize the one that matches ground-truth constraints from the task (e.g., origin city, dates).\n"
                "4. NO HALLUCINATION: You are FORBIDDEN from using names not in the 'AVAILABLE TOOLS' list.\n"
                "5. INTERACTION LOCK: Use the 'respond' tool if the intent is to 'Ask the user'.\n"
                "6. OUTPUT: JSON object with 'action' (name, arguments).\n\n"
                f"AVAILABLE TOOLS: {self.tools_info}"
            )
        else:
            self.system_prompt = (
                "You are the PEVAL Tactician. Map instruction to tool. Check 'world_snapshot' for missing parameters before asks. "
                "Output STRICTLY as a JSON object with 'name' and 'arguments'.\n"
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
