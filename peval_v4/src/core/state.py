from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class PEVState(BaseModel):
    """
    The Single Source of Truth Context Kernel.
    Strictly typed to prevent 'Graph Failed' errors.
    """
    # --- Core Memories ---
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Raw conversation history with current distilled summary."
    )
    memory_kernel: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Atomic extracted facts from API observations."
    )
    global_wisdom: List[str] = Field(default_factory=list)
    
    # --- Tactical Outputs ---
    reformulated_observation: str = "" # Used for IRMA strategy
    current_plan: str = ""
    drafted_tool_call: Dict[str, Any] = Field(default_factory=dict)
    
    # --- Verification & Loops ---
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
