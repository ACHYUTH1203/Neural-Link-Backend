from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from state import GraphState
from nodes import rag_generator_node,validator_node,web_search_node


def route_after_validation(state: GraphState):
    if state["needs_assistance"] is True:
        if state.get("revision_count", 0) < 1:
            return "web_search"
    return END

workflow = StateGraph(GraphState)

workflow.add_node("generator", rag_generator_node)
workflow.add_node("validator", validator_node)
workflow.add_node("web_search", web_search_node)

workflow.add_edge(START, "generator")
workflow.add_edge("generator", "validator")

workflow.add_conditional_edges(
    "validator",
    route_after_validation,
    {
        "web_search": "web_search",
        END: END
    }
)

workflow.add_edge("web_search", "generator")

memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "musk_fan_1"}}
    initial_input = {"query": "How do you apply the 5-step algorithm to Starship production?"}
    
    for event in app.stream(initial_input, config):
        print(event)
