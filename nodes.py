import os
from dotenv import load_dotenv
from llm import ElonLLM
from state import GraphState
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_api_key = os.getenv("TAVILY_API_KEY")
def rag_generator_node(state: GraphState):
    """Generates the initial response based on retrieved MongoDB context."""
    llm = ElonLLM() 
    
    context_text = "\n\n".join([f"Source ({d['source_collection']}): {d['content']}" for d in state["documents"]])
    
    system_prompt = """
    You are the Elon Musk Digital Twin. Your mission is to answer queries using only the provided context.
    
    LOGIC REQUIREMENTS:
    1. Apply FIRST PRINCIPLES: Break the problem down to its fundamental truths.
    2. BE CONCISE: Avoid 'fluff'. Use the style found in your 'threads' data.
    3. NO HALLUCINATION: If the context is missing specific facts, do not invent them.
    """
    
    focus_prompt = f"Using this CONTEXT:\n{context_text}\n\nAnswer the user's question."
    
    response = llm.get_response(
        system_instruction=system_prompt,
        user_query=focus_prompt,
        temperature=0.2 
    )
    
    return {"final_response": response}

class GradeSchema(BaseModel):
    score: float = Field(description="Score between 0 and 1.0 based on factual accuracy and relevance.")
    is_supported: bool = Field(description="Is the answer supported by the documents?")
    feedback: str = Field(description="Reasoning for the assigned score.")

def validator_node(state: GraphState):
    """Validates the RAG answer. If score < 0.7, triggers web search fallback."""
    llm = ElonLLM()
    structured_llm = llm.llm.with_structured_output(GradeSchema)
    
    validation_prompt = f"""
    Compare the following Answer against the provided Context for the Query.
    
    QUERY: {state['query']}
    ANSWER: {state['final_response']}
    CONTEXT: {state['documents']}
    
    CRITERIA:
    - Give a score of 1.0 if the answer is perfectly supported by the context.
    - Give a score below 0.7 if the answer is generic, hallucinated, or 'I don't know'.
    """
    grade = structured_llm.invoke(validation_prompt)
    low_score = grade.score < 0.70
    
    return {

        "rag_answer": state['final_response'] if low_score else None,

        "needs_assistance": low_score,
        "error_log": state.get("error_log", []) + [f"Grade: {grade.score} - {grade.feedback}"]
    }


def web_search_node(state: GraphState):
    """
    Fallback node: Executes a targeted web search if local MongoDB 
    data failed the 0.70 quality threshold.
    """
    search_tool = TavilySearchResults(k=5,tavily_api_key=tavily_api_key)
    
    llm = ElonLLM()
    search_query_gen_prompt = f"""
    SYSTEM INSTRUCTION:
    You are a Senior Search Engineer specializing in Elon Musk's engineering and business frameworks. 
    Your goal is to rescue a failed RAG attempt by generating a high-precision search query.

    CONTEXT OF FAILURE:
    - Original Query: "{state['query']}"
    - Knowledge Gap: {state.get('error_log', [])[-1]}

    YOUR TASK:
    Generate a single, technical search query. 
    1. USE SEARCH OPERATORS: Use "quotes" for specific terms or site: (e.g. site:x.com or site:tesla.com) if relevant.
    2. SEMANTIC EXPANSION: Use technical synonyms that Elon would use (e.g., 'mass-to-orbit' instead of 'rocket capacity').
    3. TEMPORAL AWARENESS: If the query is about recent events, include the current year 2026.
    4. DE-NOISING: Exclude common fluff. Focus on 'First Principles' and 'Engineering Truths'.

    Return ONLY the optimized query string. No explanations.
    """
    optimized_query = llm.get_response(
        system_instruction="You are a search optimization expert.",
        user_query=search_query_gen_prompt,
        temperature=0.0
    )

    print(f"Triggering Web Search for: {optimized_query}")

    raw_results = search_tool.invoke({"query": optimized_query})
    web_docs = []
    for result in raw_results:
        web_docs.append({
            "content": result["content"],
            "url": result["url"],
            "source_collection": "web_search",
            "score": 1.0 
        })
    
    return {
        "documents": state.get("documents", []) + web_docs,
        "revision_count": state.get("revision_count", 0) + 1,
        "needs_assistance": False 
    }