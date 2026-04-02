from typing import Dict, Any
from ..core.state import PEVState
from ..core.model_client import ModelClient
from ..core.logger import PEVLogger

class Strategist:
    """
    Component: Hierarchical Planner
    Role: Sets high-level objectives in natural language.
    Constraint: Does not perform tool calls directly.
    """
    def __init__(self, tools_info: list):
        # User Mandate: Use gpt-5.4-pro / gpt-4o for all intelligence
        self.client = ModelClient(mode="summarizer")
        
        # Generation: Function Mapping (Name, Inputs, Outputs/Goal)
        # Handle both flat and OpenAI-wrapped tool definitions (tau-bench compatible)
        self.function_mapping = []
        for t in tools_info:
            target = t.get("function", t) # Support OpenAI-style wrapping
            name = target.get("name", "Unknown")
            desc = target.get("description", "No description")
            params = target.get("parameters", {}).get("required", [])
            self.function_mapping.append(f"- {name}({', '.join(params)}): {desc}")
        
        self.function_mapping = "\n".join(self.function_mapping)

        self.system_prompt = (
            "You are the Lead Systems Architect in an Abstract State-Constraint Decomposition (ASCD) network.\n"
            "You MUST perform a 3-Stage Hierarchical Blackboard Refinement (HBR) based on the history and functional trace:\n\n"
            "STAGE 1: Entity/Data Scan\n"
            "Extract new facts from recent observations. Update 'state_s'. DO NOT invent values (e.g. user_id).\n\n"
            "STAGE 2: Constraint Scan\n"
            "Identify Hard limits (H) and Soft preferences (sigma). Update 'constraint_set_c'.\n\n"
            "STAGE 3: Gap Scan (gamma)\n"
            "Compare your 'state_s' against the required parameters of the target tool in the Function Map. "
            "If a parameter is MISSING, append it to 'gap_manifest_y' and instruct the Tactician to call 'respond' to ask the user.\n\n"
            "AVAILABLE TOOL MAP (Function Mapping):\n"
            f"{self.function_mapping}\n\n"
            "OUTPUT FORMAT (JSON ONLY):\n"
            "{\n"
            "  \"BlackboardSSO\": {\n"
            "    \"state_s\": {\"user_id\": \"...\", \"origin\": \"...\"},\n"
            "    \"constraint_set_c\": {\"hard\": [...], \"soft\": [...]},\n"
            "    \"gap_manifest_y\": [\"Missing Argument 1\", ...],\n"
            "    \"refined_tactical_plan\": \"Specific masked command for the deterministic Tactician (e.g. Call book_reservation with ID X or Call respond to ask for Y)\"\n"
            "  }\n"
            "}"
        )

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        PEVLogger.node("Strategist", "Abstract State-Constraint Decomposition (HBR Scans)...")
        
        from ..core.config import PEVConfig
        
        # HYBRID CONTEXT: Combine global summary with very recent raw history and functional trace
        summary_part = f"GLOBAL ARCHIVE (Summary): {state.summary}\n" if state.summary else ""
        recent_history = f"LATEST RAW TURNS: {str(state.history[-5:])}\n"
        functional_trace = f"FUNCTIONAL TRACE f(x)->y: {str(state.manifest.functional_trace[-3:])}"
        context = f"{summary_part}{recent_history}{functional_trace}"
        
        # Include feedback from Audit if the previous attempt failed
        feedback = ""
        if state.audit_feedback:
            feedback = f"\n\n[INSTRUCTION UPDATE]: {state.audit_feedback}"
            
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context: {context}\nIdentify missing entities, resolve gaps, and update the Blackboard. Output JSON."}
        ]
        
        decision_raw = self.client.chat(prompt)
        
        # Extract JSON from potential model conversational wrapper
        import json, re
        decision = {}
        json_match = re.search(r'\{.*\}', decision_raw, re.DOTALL)
        if json_match:
            try:
                decision = json.loads(json_match.group())
            except:
                pass
        
        # Format the result for the Engine updating the BlackboardSSO
        blackboard_data = decision.get("BlackboardSSO", {})
        objective = blackboard_data.get("refined_tactical_plan", "Execute next logical task step.")
        
        PEVLogger.info(f"Refined Tactical Plan: {objective[:100]}...")
        if blackboard_data.get("gap_manifest_y"):
            PEVLogger.warn(f"Gaps identified: {blackboard_data.get('gap_manifest_y')}")
        
        return {
            "strategic_instruction": objective,
            "manifest": blackboard_data, # Triggers deep merge in engine.py
            "node_logs": [{"node": "Strategist", "content": f"Tactical Plan: {objective}"}]
        }
