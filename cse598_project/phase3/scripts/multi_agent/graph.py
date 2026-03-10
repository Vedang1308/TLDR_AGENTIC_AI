from langgraph.graph import StateGraph, END
from typing import TypedDict, Callable
from .state import PevState
from .nodes import planner_node, executor_node, syntax_monitor_node, validator_node

def orchestrator_router(state: PevState) -> str:
    """
    Decides the next node after Planner.
    If the planner marked the task as completed, we finish.
    Otherwise, we send the plan to the Executor.
    """
    if state.task_completed:
        return END
    return "executor"

def syntax_router(state: PevState) -> str:
    """
    Decides where to go after the Syntax Loop Monitor.
    If rejected (malformed or looping), go back to Planner to re-assess.
    If approved, go to Validator.
    """
    if state.rejection_feedback and state.rejection_source == "syntax_monitor":
        if state.rejection_count >= 3:
            print(f"      ↳ [Router] Max internal retries reached ({state.rejection_count}). Bypassing internal loop. Failing open.")
            return END
        return "planner"
    return "validator"

def validation_router(state: PevState) -> str:
    """
    Decides where to go after Validator Critic.
    If approved, the graph ends (the tool call is sent back to the Tau-Bench Env).
    If rejected due to policy, back to Planner to re-plan action.
    """
    if state.rejection_feedback and state.rejection_source == "validator":
        if state.rejection_count >= 3:
            print(f"      ↳ [Router] Max internal retries reached ({state.rejection_count}). Bypassing internal loop. Failing open.")
            return END
        return "planner"
    # End of graph execution, return control to main env loop to execute the tool
    return END

def create_pev_graph():
    # 1. Initialize StateGraph
    workflow = StateGraph(PevState)
    
    # 2. Add Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("syntax_monitor", syntax_monitor_node)
    workflow.add_node("validator", validator_node)
    
    # 3. Define the Flow
    workflow.set_entry_point("planner")
    
    workflow.add_conditional_edges(
        "planner",
        orchestrator_router,
        {
            "executor": "executor",
            END: END
        }
    )
    
    workflow.add_edge("executor", "syntax_monitor")
    
    workflow.add_conditional_edges(
        "syntax_monitor",
        syntax_router,
        {
            "planner": "planner", # Retry planning
            "validator": "validator" # Move to logical validation
        }
    )
    
    workflow.add_conditional_edges(
        "validator",
        validation_router,
        {
            "planner": "planner", # Policy violation, try again
            END: END # Approved, return to Tau-Bench
        }
    )
    
    # Compile
    app = workflow.compile()
    return app
