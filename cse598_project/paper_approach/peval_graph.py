from langgraph.graph import StateGraph, END
from .state import PevState
from .nodes import (
    summarizer_node,
    strategist_node,
    tactician_node,
    translator_node,
    monitor_node,
    validator_node,
    learning_node
)

def create_peval_graph():
    """
    Creates the PEVAL LangGraph orchestration.
    """
    workflow = StateGraph(PevState)
    
    # 1. Define Nodes
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("tactician", tactician_node)
    workflow.add_node("translator", translator_node)
    workflow.add_node("monitor", monitor_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("learning", learning_node)
    
    # 2. Define Edges (The Flow)
    workflow.set_entry_point("summarizer")
    
    workflow.add_edge("summarizer", "strategist")
    workflow.add_edge("strategist", "tactician")
    workflow.add_edge("tactician", "translator")
    workflow.add_edge("translator", "monitor")
    
    # Conditional Edge for Monitor (Loop Detection)
    def monitor_router(state: PevState):
        if state.get("current_node") == "strategist":
            return "strategist"
        return "validator"
        
    workflow.add_conditional_edges(
        "monitor",
        monitor_router,
        {
            "strategist": "strategist", # Redirect back to planning on loop
            "validator": "validator"
        }
    )
    
    # Conditional Edge for Validator (Policy Failure or Final Dispatch)
    def validator_router(state: PevState):
        if state.get("current_node") == "strategist":
            return "strategist" # Back to planner on policy violation
        return "learning" # Forward to learning on success/final turn
        
    workflow.add_conditional_edges(
        "validator",
        validator_router,
        {
            "strategist": "strategist",
            "learning": "learning"
        }
    )
    
    workflow.add_edge("learning", END)
    
    return workflow.compile()
