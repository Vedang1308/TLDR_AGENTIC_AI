from typing import Any, Dict

from ..core.state import PEVState
from ..intelligence.strategist import Strategist
from ..intelligence.tactician import Tactician
from ..guardrails.monitor import OutcomeMonitor
from ..guardrails.translator import SemanticTranslator
from ..intelligence.auditor import Auditor
from ..memory.distiller import ContextDistiller
from ..intelligence.reformulator import InputReformulator
from ..metacognition.reflector import ErrorReflector
from ..core.config import PEVConfig
from ..core.logger import PEVLogger

class PEVEngine:
    """
    Native Python Orchestration Engine for Phase 4 Lite.
    Replaces LangGraph to eliminate graph recursion errors while keeping the exact same flow.
    """
    def __init__(self, tools_info: list, wiki: str):
        self.distiller = ContextDistiller()
        self.reformulator = InputReformulator()
        self.planner = Strategist(tools_info)
        self.tactician = Tactician(tools_info)
        self.translator = SemanticTranslator(tools_info)
        self.monitor = OutcomeMonitor()
        self.auditor = Auditor(wiki)
        self.reflector = ErrorReflector()

    def _update_state(self, state: PEVState, updates: Dict[str, Any]) -> PEVState:
        """Utility to apply node outputs to the current state."""
        state_dict = state.model_dump()
        
        for k, v in updates.items():
            if k in ["node_logs", "action_fingerprints", "history", "memory_kernel", "global_wisdom"]:
                if isinstance(v, list):
                    state_dict[k].extend(v)
                else:
                    state_dict[k].append(v)
            # DEEP MERGE for BlackboardSSO components
            elif k == "manifest" and isinstance(v, dict):
                current_manifest = state_dict.get(k, {})
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, dict) and sub_k in current_manifest:
                        current_manifest[sub_k].update(sub_v)
                    else:
                        current_manifest[sub_k] = sub_v
                state_dict[k] = current_manifest
            else:
                state_dict[k] = v
                
        return PEVState(**state_dict)

    def invoke(self, initial_state: PEVState, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Unified Predictive Macro-Execution (PME) Cycle.
        """
        state = initial_state
        recursion_limit = config.get("recursion_limit", PEVConfig.RECURSION_LIMIT) if config else PEVConfig.RECURSION_LIMIT
        
        # 1. Distiller (Extracts Variables & Summarizes History)
        distilled_data = self.distiller(state)
        state = self._update_state(state, distilled_data)
            
        # PME Execution Loop
        attempts = 0
        while attempts < recursion_limit:
            attempts += 1
            
            try:
                # Unified Architecture: PME Roadmapping
                roadmap = state.manifest.roadmap
                roadmap_progress = state.manifest.roadmap_progress
                last_memory = state.manifest.write_ahead_memory[-1] if state.manifest.write_ahead_memory else {}
                
                # RE-PLAN TRIGGER: If roadmap is empty, last action failed, or last action was interactive (respond)
                from tau_bench.types import RESPOND_ACTION_NAME
                was_respond = last_memory.get("action") == RESPOND_ACTION_NAME
                must_replan = last_memory.get("status") in ["DATA_MISSING", "ERROR"] or not roadmap or was_respond
                
                if must_replan:
                    PEVLogger.node("Strategist", "Unified Macro-Planning (HBR Scans)...")
                    planner_updates = self.planner(state)
                    state = self._update_state(state, planner_updates)
                else:
                    # Fast-Track: Follow the Roadmap
                    # Use the first Step in Roadmap that is not marked DONE in progress
                    active_step = next((s for s in roadmap if roadmap_progress.get(s) != "DONE"), roadmap[0])
                    PEVLogger.success(f"PME Fast-Track: Dispatching '{active_step[:40]}...'")
                    state.manifest.refined_tactical_plan = active_step
                
                # 4. Execute (Draft action)
                state = self._update_state(state, self.tactician(state))
            except Exception as e:
                PEVLogger.error(f"Inference Stall: {e}")
                state_dict = state.model_dump()
                state_dict["is_stalled"] = True
                return state_dict
            
            # 5. Translate
            state = self._update_state(state, self.translator(state))
            
            # 6. Monitor (Detect Stagnation/Loops)
            state = self._update_state(state, self.monitor(state))
            
            # 7. Audit (Security & Policy)
            state = self._update_state(state, self.auditor(state))
            
            # 8. Loop & Policy Enforcement
            if state.is_loop or state.policy_violation:
                state.consecutive_errors += 1
                if state.consecutive_errors >= 3 or attempts >= recursion_limit:
                    PEVLogger.error(f"Reasoning stalled (Loop/Violations). Breaking for Rollback.")
                    break
                # continue will trigger 'must_replan' in the next iteration 
                # because last_memory status will be DATA_MISSING or previous attempt failed.
                continue
            else:
                state.consecutive_errors = 0
                break
                
        # Return state as dictionary (to match the old graph.invoke return type)
        return state.model_dump()
