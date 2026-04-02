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
            "2. AGNOSTIC TOOL DISCOVERY: You MUST scan the 'AVAILABLE TOOLS' block to find the exact tool name for your intent. \n"
            "   - If your goal is to 'Find a flight/product', identify the tool that queries the catalog.\n"
            "   - If a tool returns a summary list, you MUST call the corresponding 'get_details' tool for each candidate to verify specific constraints (times, prices, stock).\n"
            "   - DO NOT assume a tool exists if it is not in the 'AVAILABLE TOOLS' list.\n"
            "3. DISCOVERY HIERARCHY: Use (A) Snapshot Scan -> (B) Tool Discovery -> (C) Customer Interaction.\n"
            "4. NO REDUNDANT QUESTIONS: Do NOT ask for information already in the snapshot or history.\n"
            "5. NO MULTI-QUESTION ROADMAPS: If the first step is 'respond', your roadmap must be length 1.\n"
            "6. REFUSAL/ESCALATION: Transfer to human if the user refuses or stalls.\n"
            "7. STRICT MATH: Only use exact numerical values provided by tools.\n"
            "8. MEMORY AUDIT: Check 'write_ahead_memory' before generating the roadmap.\n"
            "9. ROADMAPPING: Generate exactly 5 steps (unless step 1 is 'respond').\n"
            "10. IMMEDIATE TARGET: Set 'refined_tactical_plan' to the first roadmap step.\n\n"
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
        
        # HYBRID CONTEXT: Include persistent facts (Snapshot) with recent history
        summary_part = f"GLOBAL ARCHIVE (Summary): {state.summary}\n" if state.summary else ""
        world_snapshot = f"WORLD SNAPSHOT (Blackboard SSO): {str(state.manifest.world_snapshot)}\n"
        recent_history = f"LATEST RAW TURNS: {str(state.history[-5:])}\n"
        write_ahead_memory = f"WRITE-AHEAD MEMORY LOG (Checkpoint Status): {str(state.manifest.write_ahead_memory[-3:])}\n"
        functional_trace = f"FUNCTIONAL TRACE f(x)->y: {str(state.manifest.functional_trace[-3:])}"
        context = f"{summary_part}{world_snapshot}{recent_history}{write_ahead_memory}{functional_trace}"
        
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
