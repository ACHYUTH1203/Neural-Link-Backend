import datetime
import os
import logging
import uuid
import numpy as np
from dotenv import load_dotenv
from llm import ElonLLM
from state import GraphState
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from datetime import datetime


mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
mongo_db = mongo_client["Elon"]
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ElonDigitalTwin")

tavily_api_key = os.getenv("TAVILY_API_KEY")
books_col = mongo_db["books"]
frameworks_col = mongo_db["frameworks"]
podcasts_col = mongo_db["podcasts"]
threads_col = mongo_db["threads"]
user_interactions_col = mongo_db["user_interaction"]

def query_refiner_node(state: GraphState):
    logger.info("--- REFINING QUERY BASED ON HISTORY ---")
    llm = ElonLLM()

    history_cursor = user_interactions_col.find(
        {"user_id": state["user_id"]}
    ).sort("timestamp", -1).limit(5)

    history = []
    for h in history_cursor:
        h["_id"] = str(h["_id"]) 
        history.append(h)
    
    history = history[::-1]

    if not history:
        return {"query": state["query"], "chat_history": []}

    formatted_history = ""
    for h in history:

        formatted_history += f"User: {h['user_query']}\nElon: {h['response']}\n---\n"

    refinement_prompt = f"""
    You are a Query Refiner. Your job is to look at the 'Current Query' and the 'Chat History'.
    If the 'Current Query' refers to something in the history (e.g., uses 'it', 'him', 'that project'), 
    rewrite it into a standalone query that captures the full context.
    If it is a new, independent topic, return the 'Current Query' exactly as is.

    CHAT HISTORY:
    {formatted_history}

    CURRENT QUERY:
    {state['query']}

    Return ONLY the refined query text.
    """

    refined_query = llm.get_response(
        system_instruction="You are a linguistic context expert.",
        user_query=refinement_prompt,
        temperature=0
    ).strip()

    logger.info(f"Original: {state['query']} -> Refined: {refined_query}")
    
    return {
        "query": refined_query, 
        "original_query": state["query"], 
        "chat_history": history
    }

def save_interaction_node(state: GraphState):
    """Saves the turn to MongoDB so the Refiner can find it later."""
    user_interactions_col.insert_one({
        "user_id": state["user_id"],
        "user_query": state["original_query"],
        "response": state["final_response"],
        "timestamp": datetime.utcnow() 
    })
    return state

def rag_generator_node(state: GraphState):
    logger.info("--- ENTERING RAG GENERATOR NODE ---")

    llm = ElonLLM()
    query = state["query"]
    query_embedding = embedding_model.encode(query).tolist()

    logger.info("Performing vector similarity search...")
    def fetch_all(col):
        return list(col.find({}, {"embedding": 1, "content": 1, "title": 1, "book_name": 1, "orig_tweet_text": 1}))

    books = fetch_all(books_col)
    frameworks = fetch_all(frameworks_col)
    threads = fetch_all(threads_col)
    podcasts = fetch_all(podcasts_col)

    all_docs = books + frameworks + threads + podcasts

    scored_docs = []

    for doc in all_docs:
        if "embedding" not in doc:
            continue

        score = cosine_similarity(query_embedding, doc["embedding"])

        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in scored_docs[:5]]

    logger.info(f"Top {len(top_docs)} relevant documents retrieved")
    context_chunks = []

    for d in top_docs:
        if "content" in d:
            context_chunks.append(d["content"])
        elif "orig_tweet_text" in d:
            context_chunks.append(d["orig_tweet_text"])
        elif "title" in d:
            context_chunks.append(d["title"])

    if len(context_chunks) == 0:
        context_text = "NO CONTEXT AVAILABLE"
    else:
        context_text = "\n\n".join(context_chunks)
       

    system_prompt = """
    You are the Elon Musk Digital Twin.

    Use ONLY the provided context.
    Apply first-principles reasoning.
    Be concise and direct.
    """

    focus_prompt = f"""
    CONTEXT:
    {context_text}

    USER QUESTION:
    {query}
    """

    response = llm.get_response(
        system_instruction=system_prompt,
        user_query=focus_prompt,
        temperature=0.2
    )

    logger.info("Response generated successfully.")

    return {
        "final_response": response,
        "response_type":"rag"
    }

def validator_node(state: GraphState):
    """Simplified validator: Returns only a score to decide on web search fallback."""
    logger.info("--- ENTERING VALIDATOR NODE ---")
    llm = ElonLLM()
    
    validation_prompt = f"""
    Evaluate this response for the Elon Musk Digital Twin.
    
    QUERY: {state['query']}
    CONTEXT: {state.get('rag_docs', 'NO CONTEXT AVAILABLE')}
    ANSWER: {state['final_response']}

    CRITERIA:
    1. Accuracy: Is it supported by context?
    2. Persona: Does it sound like Elon (blunt, first-principles, no fluff)?

    Output ONLY a single number between 0.0 and 1.0. 
    A score < 0.7 means we must discard this and use Web Search.
    """
    
    # Get numeric response from LLM
    raw_score = llm.get_response(
        system_instruction="Output numbers only.", 
        user_query=validation_prompt,
        temperature=0
    )
    
    try:
        score = float(raw_score.strip())
    except:
        score = 0.0 # Fallback to web search on error

    low_score = score < 0.70
    
    return {
        "validation_score": score,
        "needs_assistance": low_score
    }

def web_search_node(state: GraphState):
    """
    Fallback node: Executes a targeted web search if local MongoDB 
    data failed the 0.70 quality threshold.
    """
    logger.info("--- ENTERING WEB SEARCH NODE ---")
    if not tavily_api_key:
        logger.error(" TAVILY_API_KEY is missing. Skipping web search.")
        return {
            "documents": state.get("documents", []),
            "revision_count": state.get("revision_count", 0) + 1,
            "needs_assistance": False,
            "error_log": state.get("error_log", []) + ["Web search skipped: Missing API Key"]
        }

    search_tool = TavilySearchResults(
        k=5,
        tavily_api_key=tavily_api_key,
        include_answer=False,
        include_raw_content=False
    )

    llm = ElonLLM()

    # Optimized search query generation
    search_query_gen_prompt = f"""
    Generate a highly specific search query to answer the user's question.
    User question: {state['query']}
    Return ONLY the search query text.
    """

    logger.info("Generating optimized search query...")
    optimized_query = llm.get_response(
        system_instruction="You are a world-class search optimization expert.",
        user_query=search_query_gen_prompt,
        temperature=0.0
    ).strip().replace('"', '') # Clean quotes

    logger.info(f"Final Search Query Used: '{optimized_query}'")

    web_docs = []
    try:
        logger.info("Executing Tavily API search...")
        raw_results = search_tool.invoke({"query": optimized_query})
        
        results = raw_results if isinstance(raw_results, list) else raw_results.get("results", [])
        
        seen_urls = set()
        for result in results:
            content = result.get("content") or result.get("snippet") or ""
            url = result.get("url", "")
            if content and url not in seen_urls:
                seen_urls.add(url)
                web_docs.append({"content": content.strip(), "url": url})

        # --- REFINED ELON PERSONA PROMPT ---
        system_prompt = f"""
        You are the Elon Musk Digital Twin. You are high-signal, physics-first, and extremely direct.

        STRICT OPERATIONAL DIRECTIVES:
        1. NO AI PREAMBLE: Never start with "Based on the search results" or "According to." Start with the answer.
        2. NO FLUFF: Delete words like "prominent," "growing importance," or "expected to boom." Use data and facts.
        3. PHYSICS-FIRST: If the query is technical, frame the answer in terms of energy, materials, or engineering constraints.
        4. PERSONA: You are blunt. If a trend is hype, call it out. Use short sentences. 
        5. LINGUISTIC STYLE: Use terms like "high signal," "first principles," "vector," or "orders of magnitude" only if they fit the logic.

        WEB SEARCH CONTEXT:
        {web_docs}
        """

        focus_prompt = f"""
        USER QUESTION: {state['query']}

        Provide the final answer now. Zero fluff. Just high-signal engineering and business logic.
        """

        web_response = llm.get_response(
            system_instruction=system_prompt,
            user_query=focus_prompt,
            temperature=0.3 # Slightly higher for more natural "Elon" phrasing
        )

        return {
            "final_response": web_response,
            "needs_assistance": False,
            "revision_count": state.get("revision_count", 0) + 1,
            "web_results": web_docs,
            "error_log": state.get("error_log", []) + ["Web search fallback executed successfully"]
        }

    except Exception as e:
        logger.error(f"Web search failed: {str(e)}")
        return {
            "needs_assistance": False,
            "revision_count": state.get("revision_count", 0) + 1,
            "error_log": state.get("error_log", []) + [f"Web search error: {str(e)}"]
        }