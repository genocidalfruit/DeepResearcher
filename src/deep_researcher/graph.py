from langgraph.graph import StateGraph, END
from deep_researcher.agents import (
    ResearchState,
    manager_agent,
    search_agent,
    writer_agent,
    critique_agent
)

def should_continue(state: ResearchState) -> str:
    """Takes Critique feedback and decides if report is final or needs rewrite."""
    valid = state.get("is_valid", False)
    rev_count = state.get("revision_count", 0)
    
    print(f"\n[Router] revision_count: {rev_count}, valid: {valid}")
    if valid:
        print("[Router] Draft is accepted!")
        return END
    elif rev_count >= 3:
        print("[Router] Maximum revisions reached, forcing completion.")
        return END
    else:
        print("[Router] Draft rejected. Routing back to Writer Agent.")
        return "writer"

def build_graph():
    """Builds the deep researcher workflow using LangGraph."""
    workflow = StateGraph(ResearchState)
    
    # Define Agents
    workflow.add_node("manager", manager_agent)
    workflow.add_node("search", search_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("critique", critique_agent)
    
    # Layout processing pipeline
    workflow.set_entry_point("manager")
    
    workflow.add_edge("manager", "search")
    workflow.add_edge("search", "writer")
    workflow.add_edge("writer", "critique")
    
    workflow.add_conditional_edges(
        "critique",
        should_continue,
        {
            END: END,
            "writer": "writer"
        }
    )
    
    return workflow.compile()
