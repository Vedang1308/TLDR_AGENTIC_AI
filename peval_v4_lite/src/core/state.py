from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class SystemManifest(BaseModel):
    """
    Formal Problem Definition: P = <S, G, A, C>
    Transfers 'Instructional Noise' into 'Systemic Constraints'.
    """
    state_s: Dict[str, Any] = Field(default_factory=dict, description="Current knowledge (User ID, context).")
    goal_g: str = Field(default="", description="The final outcome (e.g., Flight Booked).")
    actions_a: List[str] = Field(default_factory=list, description="List of tools deemed necessary for this path.")
    constraints_c: Dict[str, Any] = Field(default_factory=list, description="Hard and Soft constraints (Time, Price, Style).")

class PEVState(BaseModel):
    """
    The Single Source of Truth Context Kernel.
    Strictly typed to prevent 'Graph Failed' errors.
    """
    model_config = ConfigDict(extra='allow')
    
    # --- The P=<S,G,A,C> Manifest ---
    manifest: SystemManifest = Field(default_factory=SystemManifest)

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
    persistent_ner: Dict[str, Any] = Field(default_factory=dict)
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
    is_stalled: bool = False
    
    # --- Execution Result ---
    last_observation: str = ""
    task_completed: bool = False
    reward: float = 0.0
    
    # --- Node Internal Logs (for debugging) ---
    node_logs: List[Dict[str, str]] = Field(default_factory=list)
