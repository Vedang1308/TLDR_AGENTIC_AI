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
            "You are the Lead Systems Coordinator operating in Predictive Macro-Execution (PME) mode.\n"
            "Your goal is to maintain the Blackboard SSO and generate a high-speed Tactical Roadmap.\n\n"
            "INSTRUCTIONS:\n"
            "1. WORLD SNAPSHOT: Update 'world_snapshot' with discovered Entities (S), Constraints (C), and Data Gaps (G).\n"
            "2. MEMORY AUDIT: Check 'write_ahead_memory'. If status is [DATA_MISSING] or [ERROR], you MUST PIVOT in the roadmap. Do not repeat failed steps.\n"
            "3. ROADMAPPING: Generate a 'roadmap' of EXACTLY 5 upcoming internal tactical steps (e.g., ['1. Search JFK', '2. Compare Prices', ...]).\n"
            "4. IMMEDIATE TARGET: Set 'refined_tactical_plan' to the first step of your roadmap.\n\n"
            "AVAILABLE TOOLS:\n"
            f"{self.function_mapping}\n\n"
            "OUTPUT FORMAT (JSON ONLY):\n"
            "{\n"
            "  \"extended_thinking\": \"Hidden reasoning trace.\",\n"
            "  \"BlackboardSSO\": {\n"
            "    \"world_snapshot\": {\"entities\": {...}, \"constraints\": {...}, \"gaps\": [...]},\n"
            "    \"roadmap\": [\"1. Step A\", \"2. Step B\", \"3. Step C\", \"4. Step D\", \"5. Step E\"],\n"
            "    \"roadmap_progress\": {\"1. Step A\": \"TODO\", ...},\n"
            "    \"refined_tactical_plan\": \"Immediate action string for the Tactician.\"\n"
            "  }\n"
            "}"
        )

    def __call__(self, state: PEVState) -> Dict[str, Any]:
        PEVLogger.node("Strategist", "Abstract State-Constraint Decomposition (HBR Scans)...")
        
        from ..core.config import PEVConfig
        
        # HYBRID CONTEXT: Combine global summary with very recent raw history and memory ledger
        summary_part = f"GLOBAL ARCHIVE (Summary): {state.summary}\n" if state.summary else ""
        recent_history = f"LATEST RAW TURNS: {str(state.history[-5:])}\n"
        write_ahead_memory = f"WRITE-AHEAD MEMORY LOG (Checkpoint Status): {str(state.manifest.write_ahead_memory[-3:])}\n"
        functional_trace = f"FUNCTIONAL TRACE f(x)->y: {str(state.manifest.functional_trace[-3:])}"
        context = f"{summary_part}{recent_history}{write_ahead_memory}{functional_trace}"
        
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
        
        PEVLogger.info(f"CEL Extended Thinking: {decision.get('extended_thinking', 'None')[:150]}...")
        if decision.get('re_plan', False):
            PEVLogger.warn("CEL Reconciliation Triggered: RE-PLAN INITIATED")
            
        PEVLogger.info(f"Refined Tactical Plan: {objective[:100]}...")
        if blackboard_data.get("gap_manifest_y"):
            PEVLogger.warn(f"Gaps identified: {blackboard_data.get('gap_manifest_y')}")
        
        return {
            "strategic_instruction": objective,
            "manifest": blackboard_data, # Triggers deep merge in engine.py
            "node_logs": [{"node": "Strategist", "content": f"Tactical Plan: {objective[:50]}..."}]
        }
