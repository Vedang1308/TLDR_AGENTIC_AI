from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class PevState(BaseModel):
    """
    Simplified State for Phase 3 Gaudi-Lite (No LangGraph).
    Maintains the successful 'Persistence' and 'Metacognition' fields.
    """
    # History with the user
    user_conversation: List[Dict[str, str]] = Field(default_factory=list)
    
    # The Planner's current high-level plan and reasoning
    current_plan: str = Field(default="")
    
    # State flags
    task_completed: bool = Field(default=False)
    
    # Latest Tool Draft from Executor
    drafted_tool_call: Optional[Dict[str, Any]] = Field(default=None)
    
    # Feedback from Monitor or Validator (if rejected)
    rejection_feedback: Optional[str] = Field(default=None)
    rejection_source: Optional[str] = Field(default=None)
    
    # Memory Kernel: observations and API returns
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Track internal loops to prevent recursion crashes
    internal_retry_count: int = Field(default=0)
    
    # Logs for debugging
    node_logs: List[Dict[str, Any]] = Field(default_factory=list)

    # API schemas injected at strategy startup
    tools_info: List[Dict[str, Any]] = Field(default_factory=list)
    
    # --- SELF-CORRECTION FIELDS ---
    global_wisdom: List[str] = Field(default_factory=list)
    tools_wiki: str = "" 
    
    # Tracks ALL failed strategies
    failure_log: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Counts how many consecutive API errors happened
    consecutive_error_count: int = Field(default=0)
    
    # The output of the Error Reflection node
    error_reflection: Optional[str] = Field(default=None)
    
    # Path to the shared wisdom file
    wisdom_file: str = Field(default="results/phase3/persistent_wisdom.json")
