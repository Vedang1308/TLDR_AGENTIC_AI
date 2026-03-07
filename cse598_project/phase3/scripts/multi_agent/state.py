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
    
    # Logs for debugging
    node_logs: Annotated[List[Dict[str, Any]], append_to_list] = Field(default_factory=list)

