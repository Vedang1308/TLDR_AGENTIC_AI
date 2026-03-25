from typing import Annotated, TypedDict, List, Dict, Any, Union
from langgraph.graph.message import add_messages

class PevState(TypedDict):
    # Core Trajectory
    messages: Annotated[list, add_messages]
    user_utterance: str
    env_observation: str
    
    # PEVAL Modular Components
    strategic_kernel: str      # Output of Summarizer
    strategic_plan: str        # Output of Strategist
    action_draft: Dict[str, Any] # Output of Tactician
    normalized_action: Dict[str, Any] # Output of Translator
    
    # Global Knowledge
    global_wisdom: List[str]
    
    # Control Flow
    metadata: Dict[str, Any]
    retry_count: int
    current_node: str
    is_finished: bool
    reward: float
