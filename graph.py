# from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import InMemorySaver
# from state import GraphState
# from nodes import rag_generator_node,validator_node,web_search_node


# def route_after_validation(state: GraphState):
#     if state["needs_assistance"] is True:
#         if state.get("revision_count", 0) < 1:
#             return "web_search"
#     return END

# workflow = StateGraph(GraphState)

# workflow.add_node("generator", rag_generator_node)
# workflow.add_node("validator", validator_node)
# workflow.add_node("web_search", web_search_node)

# # Step 1: Always generate first
# workflow.add_edge(START, "generator")

# # Step 2: Always validate the RAG output
# workflow.add_edge("generator", "validator")

# # Step 3: Conditional Routing
# workflow.add_conditional_edges(
#     "validator",
#     route_after_validation,
#     {
#         "web_search": "web_search", # If score < 0.7
#         END: END                    # If score >= 0.7
#     }
# )

# # Step 4: Web search is the FINAL step
# workflow.add_edge("web_search", END) 


# memory = InMemorySaver()
# app = workflow.compile(checkpointer=memory)

# if __name__ == "__main__":
#     config = {"configurable": {"thread_id": "musk_fan_1"}}
    
#     # Fill all keys defined in GraphState to avoid KeyErrors in nodes
#     initial_input = {
#         "query": "How do you apply the 5-step algorithm to Starship production?",
#         "documents": [], # Initialize as empty so rag_generator_node doesn't crash
#         "revision_count": 0,
#         "error_log": [],
#         "needs_assistance": False
#     }
    
#     # Use app.invoke for a single result, or keep app.stream to see the node transitions
#     final_state = app.invoke(initial_input, config)
#     print("\n--- FINAL RESPONSE ---")
#     print(final_state.get("final_response"))


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from state import GraphState
from nodes import (
    query_refiner_node, 
    rag_generator_node, 
    validator_node, 
    web_search_node, 
    save_interaction_node
)

def route_after_validation(state: GraphState):
    """
    Determines if we need a web search fallback based on the validator score.
    """
    if state.get("needs_assistance") is True:
        # Prevent infinite loops: only allow 1 web search revision
        if state.get("revision_count", 0) < 1:
            return "web_search"
    
    # If valid or revision limit reached, save to DB
    return "save_memory"

# Initialize Graph
workflow = StateGraph(GraphState)

# Define Nodes
workflow.add_node("refiner", query_refiner_node)       # New: Resolves context
workflow.add_node("generator", rag_generator_node)
workflow.add_node("validator", validator_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("save_memory", save_interaction_node) # New: Persists to MongoDB

# --- Graph Logic ---

# 1. Start with Query Refinement
workflow.add_edge(START, "refiner")

# 2. Pass the refined query to the RAG Generator
workflow.add_edge("refiner", "generator")

# 3. Validate the output
workflow.add_edge("generator", "validator")

# 4. Decide: Web Search vs. Save & Finish
workflow.add_conditional_edges(
    "validator",
    route_after_validation,
    {
        "web_search": "web_search",
        "save_memory": "save_memory"
    }
)

# 5. After Web Search, go to Save Memory
workflow.add_edge("web_search", "save_memory")

# 6. Final step is always saving the interaction
workflow.add_edge("save_memory", END)

# Compile with persistence
memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    import uuid
    # Mocking a user session
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