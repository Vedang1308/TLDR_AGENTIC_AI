from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class PEVState(BaseModel):
    """
    The Single Source of Truth Context Kernel.
    Strictly typed to prevent 'Graph Failed' errors.
    """
    model_config = ConfigDict(extra='allow') # Resilience: don't crash if extra nodes add metadata

    # --- Core Memories ---
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Raw conversation history with current distilled summary."
    )
    summary: str = "" # Distilled version of history
    memory_kernel: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Atomic extracted facts from API observations."
    )
    global_wisdom: List[str] = Field(default_factory=list)
    
    # --- Tactical Outputs ---
    reformulated_observation: str = "" # Used for IRMA strategy
    strategic_instruction: str = ""    # Output from Strategist
    current_action_draft: Dict[str, Any] = Field(default_factory=dict) # NEW: Tactical draft
    
    # --- Verification & Loops ---
    action_fingerprints: List[str] = Field(default_factory=list)
    is_loop: bool = False
    policy_violation: Optional[str] = None
    audit_feedback: str = ""
    consecutive_errors: int = 0
    
    # --- Execution Result ---
    last_observation: str = ""
    task_completed: bool = False
    reward: float = 0.0
    
    # --- Node Internal Logs (for debugging) ---
    node_logs: List[Dict[str, str]] = Field(default_factory=list)
