from langgraph.graph import StateGraph, END
from ..core.state import PEVState
from ..intelligence.strategist import Strategist
from ..intelligence.tactician import Tactician
from ..guardrails.monitor import OutcomeMonitor
from ..guardrails.translator import SemanticTranslator
from ..intelligence.auditor import Auditor
from ..memory.distiller import ContextDistiller
from ..intelligence.reformulator import InputReformulator
from ..metacognition.reflector import ErrorReflector
from ..core.config import PEVConfig

def create_peval_graph(tools_info: list, wiki: str):
    """
    The Architecture Blueprint.
    Connects the 13 modular components into a resilient DAG.
    Dynamically routes based on TOOL_STRATEGY.
    """
    workflow = StateGraph(PEVState)

    # Initialize Components
    distiller = ContextDistiller()
    reformulator = InputReformulator()
    planner = Strategist()
    tactician = Tactician(tools_info)
    translator = SemanticTranslator(tools_info)
    monitor = OutcomeMonitor()
    auditor = Auditor(wiki)
    reflector = ErrorReflector()

    # Add Nodes
    workflow.add_node("distill", distiller)
    workflow.add_node("reformulate", reformulator)
    workflow.add_node("plan", planner)
    workflow.add_node("execute", tactician)
    workflow.add_node("translate", translator)
    workflow.add_node("monitor", monitor)
    workflow.add_node("audit", auditor)
    workflow.add_node("reflect", reflector)

    # Define the Flow based on strategy
    workflow.set_entry_point("distill")
    
    if PEVConfig.TOOL_STRATEGY == "irma":
        workflow.add_edge("distill", "reformulate")
        workflow.add_edge("reformulate", "plan")
    else:
        workflow.add_edge("distill", "plan")
        
    workflow.add_edge("plan", "execute")
    workflow.add_edge("execute", "translate")
    workflow.add_edge("translate", "monitor")
    
    # Conditional Edge: The "Determinism" Loop
    def routing_logic(state: PEVState):
        if state.is_loop:
            return "reflect" if PEVConfig.TOOL_STRATEGY == "reflection" else "plan"
        return "audit"

    workflow.add_conditional_edges("monitor", routing_logic)
    
    # Final Audit -> Dispatch (END of internal graph, back to Env)
    def audit_logic(state: PEVState):
        if state.policy_violation:
            return "reflect" if PEVConfig.TOOL_STRATEGY == "reflection" else "plan"
        return END

    workflow.add_conditional_edges("audit", audit_logic)
    
    # If reflection strategy is active, the reflector always sends it back to the planner
    workflow.add_edge("reflect", "plan")
    
    return workflow.compile()
