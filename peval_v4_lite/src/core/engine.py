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
        self.planner = Strategist()
        self.tactician = Tactician(tools_info)
        self.translator = SemanticTranslator(tools_info)
        self.monitor = OutcomeMonitor()
        self.auditor = Auditor(wiki)
        self.reflector = ErrorReflector()

    def _update_state(self, state: PEVState, updates: Dict[str, Any]) -> PEVState:
        """Utility to apply node outputs to the current state."""
        state_dict = state.model_dump()
        
        for k, v in updates.items():
            # For lists like node_logs, action_fingerprints, history, memory_kernel
            if k in ["node_logs", "action_fingerprints", "history", "memory_kernel", "global_wisdom"]:
                if isinstance(v, list):
                    state_dict[k].extend(v)
                else:
                    state_dict[k].append(v)
            # DEEP MERGE for persistent_ner so we don't lose old ingredients
            elif k == "persistent_ner" and isinstance(v, dict):
                state_dict[k].update(v)
            else:
                state_dict[k] = v
                
        return PEVState(**state_dict)

    def invoke(self, initial_state: PEVState, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Executes the agent's modular cycle for one turn.
        
        Flow:
        Distill -> [Reformulate] -> Plan -> Execute -> Translate -> Monitor -> Audit
        If Audit rejects -> [Reflect] -> loop back to Plan.
        If loop limit exceeded -> break.
        """
        state = initial_state
        recursion_limit = config.get("recursion_limit", PEVConfig.RECURSION_LIMIT) if config else PEVConfig.RECURSION_LIMIT
        
        # 1. Distiller
        state = self._update_state(state, self.distiller(state))
        
        # 2. Reformulator (If IRMA)
        if PEVConfig.TOOL_STRATEGY == "irma":
            state = self._update_state(state, self.reformulator(state))
            
        # Planner -> Auditor Replanning Loop
        attempts = 0
        while attempts < recursion_limit:
            attempts += 1
            
            # 3. Plan
            state = self._update_state(state, self.planner(state))
            
            # 4. Execute (Draft action)
            state = self._update_state(state, self.tactician(state))
            
            # 5. Translate
            state = self._update_state(state, self.translator(state))
            
            # 6. Monitor
            state = self._update_state(state, self.monitor(state))
            
            # Stagnation Break & Rollback Logic
            if state.is_loop:
                PEVLogger.warn("Monitor detected a loop. Forcing Strategic Pivot.")
                state = self._update_state(state, {
                    "is_loop": False,
                    "policy_violation": "REJECTED_STAGNATION",
                    "consecutive_errors": state.consecutive_errors + 1,
                    "audit_feedback": "Progress checked. Redundant action detected (you already did this). PIVOT requested: Do not panic or apologize. Cautiously consult your CHECKLIST and use your flexibility (e.g., try a different search or ask the user) to steadily progress the task."
                })
                if state.consecutive_errors >= 3:
                    PEVLogger.error("Inner reasoning loop stalled. Breaking for Rollback.")
                    break
                continue
                
            # 7. Audit
            state = self._update_state(state, self.auditor(state))
            
            # Evaluation
            if state.policy_violation:
                state.consecutive_errors += 1
                if state.consecutive_errors >= 3 or attempts >= recursion_limit:
                    PEVLogger.error(f"Reasoning stalled after {attempts} attempts. Breaking for Rollback.")
                    break
                    
                if PEVConfig.TOOL_STRATEGY == "reflection":
                    state = self._update_state(state, self.reflector(state))
                # Loop back to Plan
                continue
            else:
                # Validated successfully - Reset the internal error counter
                state.consecutive_errors = 0
                break
                
        # Return state as dictionary (to match the old graph.invoke return type)
        return state.model_dump()
