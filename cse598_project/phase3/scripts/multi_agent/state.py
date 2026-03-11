from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
import operator

def append_to_list(existing: List, new: Any) -> List:
    if existing is None:
        existing = []
    if isinstance(new, list):
        return existing + new
    else:
        return existing + [new]

class PevState(BaseModel):
    """
    The shared state passed around the nodes in LangGraph.
    
    Architecture Upgrade: Added failure_log and consecutive_error_count for
    domain-agnostic self-correction. The system can now track which approaches
    have already been tried and failed, and trigger deeper reflection when
    the environment repeatedly rejects actions.
    """
    # History with the user
    user_conversation: Annotated[List[Dict[str, str]], append_to_list] = Field(default_factory=list)
    
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
    memory: Annotated[List[Dict[str, Any]], append_to_list] = Field(default_factory=list)
    
    # Track internal loops to prevent recursion crashes
    internal_retry_count: int = Field(default=0)
    
    # Logs for debugging
    node_logs: Annotated[List[Dict[str, Any]], append_to_list] = Field(default_factory=list)

    # API schemas injected at strategy startup
    tools_info: List[Dict[str, Any]] = Field(default_factory=list)
    
    # --- SELF-CORRECTION FIELDS ---
    
    # Persistent Wisdom: Technical insights gathered from across all previous trials/domains
    global_wisdom: Annotated[List[str], operator.add] = []
    tools_wiki: str = "" # List of available tools and their descriptions for the Planner
    
    # Tracks ALL failed strategies as {"action": ..., "args": ..., "error": ..., "reflection": ...}
    # This is the agent's "experience log" — never reset, always growing, letting it learn
    # within a single conversation what approaches do NOT work so it can try something different.
    failure_log: Annotated[List[Dict[str, Any]], append_to_list] = Field(default_factory=list)
    
    # Counts how many consecutive API errors happened (reset on success).
    # When this hits >= 2, triggers the Error Reflection node for deeper analysis.
    consecutive_error_count: int = Field(default=0)
    
    # The output of the Error Reflection node — a synthesized diagnosis and corrective plan
    # that the Planner reads as high-priority context on its next invocation.
    error_reflection: Optional[str] = Field(default=None)

