from typing import List, Dict, Any, Optional, Annotated, TypedDict
import operator

class PevState(TypedDict):
    # Core Tau-Bench state
    messages: Annotated[List[Dict[str, Any]], operator.add]
    user_utterance: str
    env_observation: str
    
    # PEVAL specific intermediate states
    strategic_kernel: str    # Output of Summarizer
    strategic_plan: str      # Output of Strategist
    action_draft: Dict[str, Any] # Output of Tactician
    normalized_action: Dict[str, Any] # Output of Translator
    
    # Knowledge / Memory
    global_wisdom: List[str]
    metadata: Dict[str, Any]
    
    # Control flow
    retry_count: int
    current_node: str
    is_finished: bool
    reward: Optional[float]
