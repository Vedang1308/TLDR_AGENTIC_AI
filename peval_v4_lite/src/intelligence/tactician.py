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
                "You are the Deterministic Tactician (MAPPING_ENGINE). Your ONLY OUTPUT must be a valid JSON object.\n\n"
                "### CRITICAL CONSTRAINTS:\n"
                "1. NO CONVERSATIONAL FILLER. Do not say 'Certainly', 'However', or 'Here is the plan'.\n"
                "2. NO META-COMMENTARY. Do not explain why you are choosing a tool or acknowledge previous errors.\n"
                "3. SCHEMA VALIDATION: Map the Instruction to 'AVAILABLE TOOLS'.\n"
                "4. VARIABLE INJECTION: Scan the 'world_snapshot' for missing parameters. If found, use them automatically.\n"
                "5. OUTPUT FORMAT: A single JSON block. Use ```json ... ``` tags if needed, but NOTHING ELSE.\n\n"
                f"AVAILABLE TOOLS: {self.tools_info}"
            )
        else:
            self.system_prompt = (
                "STRICT JSON MAPPING ENGINE. Output only JSON. No chat.\n"
                f"Tools: {self.tools_info}"
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
            else:
                drafted_call = parsed_json

            # TYPE-SAFETY SHIELD: Ensure drafted_call is a dict
            if isinstance(drafted_call, str):
                PEVLogger.warn(f"Tactician returned string '{drafted_call}'. Converting to dict.")
                drafted_call = {"name": drafted_call, "arguments": {}}
                
            node_log_content = f"Drafted: {drafted_call}"
            PEVLogger.info(f"Action: {drafted_call}")
                
        except Exception as e:
            drafted_call = {"error": f"Failed to parse tactician output: {str(e)}", "raw": response}
            node_log_content = f"Failed Parse: {response}"
            PEVLogger.error(f"Failed Parse: {response}")

        return {
            "current_action_draft": drafted_call,
            "node_logs": [{"node": "Tactician", "content": node_log_content}]
        }
