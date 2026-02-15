from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from state import GraphState
from nodes import (
    query_refiner_node,
    conversation_strategy_node,
    rag_generator_node,
    validator_node,
    web_search_node,
    save_interaction_node
)

def route_after_validation(state: GraphState):
    if state.get("needs_assistance") is True:
        if state.get("revision_count", 0) < 1:
            return "web_search"
    return "save_memory"

def route_after_strategy(state: GraphState):
    if state.get("conversation_mode") == "clarify":
        return END
    return "generator"

workflow = StateGraph(GraphState)

workflow.add_node("refiner", query_refiner_node)
workflow.add_node("conversation_strategy", conversation_strategy_node)
workflow.add_node("generator", rag_generator_node)
workflow.add_node("validator", validator_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("save_memory", save_interaction_node)

workflow.add_edge(START, "refiner")
workflow.add_edge("refiner", "conversation_strategy")

workflow.add_conditional_edges(
    "conversation_strategy",
    route_after_strategy,
    {
        "generator": "generator",
        END: END
    }
)

workflow.add_edge("generator", "validator")

workflow.add_conditional_edges(
    "validator",
    route_after_validation,
    {
        "web_search": "web_search",
        "save_memory": "save_memory"
    }
)

workflow.add_edge("web_search", "save_memory")
workflow.add_edge("save_memory", END)

memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    import uuid

    user_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": user_id}}

    initial_input = {
        "query": "How do you apply the 5-step algorithm?",
        "user_id": user_id,
        "revision_count": 0,
        "error_log": [],
        "needs_assistance": False
    }

    print(f"--- RUNNING GRAPH FOR USER: {user_id} ---")
    final_state = app.invoke(initial_input, config)

    print("\n--- FINAL RESPONSE ---")
    print(final_state.get("final_response"))
