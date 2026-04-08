import json
import re
import os
from typing import Dict, Any
from peval_v4_lite.src.core.model_client import ModelClient
from .state import PevState

class ContextDistiller:
    """
    Component: Semantic Context Distiller / Summarizer (Step 3 in Diagram)
    Role: Uses the LOCAL USER SIMULATOR (Port 8225) to compress history.
    Reason: Offloads heavy thinking tasks from the primary Agent HPU.
    """
    def __init__(self):
        # We explicitly target the 'user' mode to use Port 8225/Qwen2.5-72B-Simulator
        self.client = ModelClient(mode="user")
        self.system_prompt = (
            "You are a Context Distiller. Your goal is to compress history into a dense 'Strategic Kernel'. \n"
            "0. INITIAL FACT-ANCHORING: Extract all IDs, names, dates, and preferences provided in the instruction and label them as 'GROUND TRUTH'. \n"
            "1. WORLD SNAPSHOT: List unique IDs (user_id, reservation_id, etc.) and associated values found in memory. \n"
            "2. SUMMARY: A 2-3 sentence summary of the current progress. \n"
            "Remove conversational fluff. Output must be perfectly clear for a Planner."
        )

    def __call__(self, state: PevState) -> Dict[str, Any]:
        """
        Processes current state.memory and user_conversation to update the kernel.
        Runs on Port 8225.
        """
        if not state.memory and not state.user_conversation:
            return {"summary": "", "world_snapshot": {}}

        # 1. Structured Variable Extraction
        extraction_prompt = [
            {"role": "system", "content": "You are an Advanced Variable Extraction Engine. Scan the history and ARCHIVE all discovered 'Task Variables' into a JSON dictionary {}. ONLY OUTPUT RAW JSON."},
            {"role": "user", "content": f"History to scan: {state.memory[-5:]}"}
        ]
        
        extracted_raw = self.client.chat(extraction_prompt)
        extracted_vars = {}
        json_match = re.search(r'\{.*\}', extracted_raw, re.DOTALL)
        if json_match:
            try:
                extracted_vars = json.loads(json_match.group())
            except: pass

        # 2. History-level distillation (Summary) with safety truncation
        def safe_truncate(obj, limit=1000):
            s = str(obj)
            return s[:limit] + "... [TRUNCATED]" if len(s) > limit else s

        recent_msgs = [safe_truncate(m) for m in state.user_conversation[-3:]]
        recent_mem = [safe_truncate(m) for m in state.memory[-10:]]
        history_str = "\n".join(recent_msgs) + "\n" + "\n".join(recent_mem)
        
        if len(history_str) > 2000:
            distill_prompt = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"History to distill:\n{history_str}"}
            ]
            summary = self.client.chat(distill_prompt)
        else:
            summary = history_str

        return {
            "summary": summary,
            "world_snapshot": extracted_vars
        }
