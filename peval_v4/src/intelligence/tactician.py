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
                "You are the PEVAL Tactician. Your role is to fulfill the Strategic Instruction "
                "using the provided tools. "
                "You must use the ReAct format: output your reasoning inside a 'thought' field, "
                "followed by the tool call in an 'action' field containing 'name' and 'arguments'. "
                "Output STRICTLY as a JSON object: {\"thought\": \"...\", \"action\": {\"name\": \"...\", \"arguments\": {...}}}.\n"
                f"Tools available: {self.tools_info}"
            )
        else:
            self.system_prompt = (
                "You are the PEVAL Tactician. Your role is to fulfill the Strategic Instruction "
                "provided by the Strategist using the following tools. "
                "Output your response STRICTLY as a JSON object with 'name' and 'arguments'.\n"
                f"Tools available: {self.tools_info}"
            )

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        PEVLogger.node("Tactician", "Drafting technical action...")
        
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Strategist's Instruction: {state.strategic_instruction}"}
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
                thought = parsed_json.get("thought", "")
                node_log_content = f"Thought: {thought} | Drafted: {drafted_call}"
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
