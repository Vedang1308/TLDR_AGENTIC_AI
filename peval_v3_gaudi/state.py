from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class PevState(BaseModel):
    """
    Standard PEV State ported from Phase 3.
    Upgraded for 100% Diagram Compliance (Strategic Kernel + World Snapshot).
    """
    # History with the user
    user_conversation: List[Dict[str, str]] = Field(default_factory=list)
    
    # --- STRATEGIC HUB (Step 3 in Diagram) ---
    strategic_kernel: str = Field(default="")
    world_snapshot: Dict[str, Any] = Field(default_factory=dict)
    
    # Latest Tool Draft from Executor
    drafted_tool_call: Optional[Dict[str, Any]] = Field(default=None)
    
    # Feedback from Monitor or Validator (if rejected)
    rejection_feedback: Optional[str] = Field(default=None)
    rejection_source: Optional[str] = Field(default=None)
    
    # Memory Kernel: observations and API returns
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    
    # The Planner's current high-level plan and reasoning
    current_plan: str = Field(default="")
    task_completed: bool = Field(default=False)
    
    # Track progress for Airline/Retail policy adherence
    user_identified: bool = Field(default=False)
    
    # Track internal loops to prevent recursion crashes
    internal_retry_count: int = Field(default=0)
    
    # Logs for debugging
    node_logs: List[Dict[str, Any]] = Field(default_factory=list)

    # API schemas injected at strategy startup
    tools_info: List[Dict[str, Any]] = Field(default_factory=list)
    
    # --- SELF-CORRECTION FIELDS (Step 9/10 Learning Node) ---
    global_wisdom: List[str] = Field(default_factory=list)
    tools_wiki: str = "" 
    
    # Experience Log
    failure_log: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Error Reflection
    consecutive_error_count: int = Field(default=0)
    error_reflection: Optional[str] = Field(default=None)
    
    # Global Repetition Tracker (Task-wide)
    tool_attempts: Dict[str, int] = Field(default_factory=dict)
    
    current_time: str = ""
