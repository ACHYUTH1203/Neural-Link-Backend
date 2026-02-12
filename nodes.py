import os
import logging
import numpy as np
from dotenv import load_dotenv
from llm import ElonLLM
from state import GraphState
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

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
        "final_response": response
    }

class GradeSchema(BaseModel):
    score: float = Field(description="Score between 0 and 1.0 based on factual accuracy and relevance.")
    is_supported: bool = Field(description="Is the answer supported by the documents?")
    feedback: str = Field(description="Reasoning for the assigned score.")

def validator_node(state: GraphState):
    """Validates the RAG answer. If score < 0.7, triggers web search fallback."""
    logger.info("--- ENTERING VALIDATOR NODE ---")
    llm = ElonLLM()
    structured_llm = llm.llm.with_structured_output(GradeSchema)
    
    validation_prompt = f"""
    Compare the following Answer against the provided Context for the Query.
    
    QUERY: {state['query']}
    ANSWER: {state['final_response']}
    
    CRITERIA:
    - Give a score of 1.0 if the answer is perfectly supported by the context.
    - Give a score below 0.7 if the answer is generic, hallucinated, or 'I don't know'.
    """
    
    logger.info("Invoking LLM Judge for validation...")
    grade = structured_llm.invoke(validation_prompt)
    
    low_score = grade.score < 0.70
    logger.info(f"Validation Result -> Score: {grade.score} | Supported: {grade.is_supported}")
    logger.info(f"Feedback: {grade.feedback}")

    if low_score:
        logger.warning(f"Quality threshold not met ({grade.score} < 0.7). Marking for assistance.")
    else:
        logger.info("Quality threshold met. Proceeding to final delivery.")
    
    return {
        "rag_answer": state['final_response'] if low_score else None,
        "needs_assistance": low_score,
        "error_log": state.get("error_log", []) + [f"Grade: {grade.score} - {grade.feedback}"],
        "validation_score": grade.score
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
            "needs_assistance": False
        }
    search_tool = TavilySearchResults(
        k=5,
        tavily_api_key=tavily_api_key,
        include_answer=False,
        include_raw_content=False
    )

    llm = ElonLLM()

    last_error = (
        state.get("error_log", [])[-1]
        if state.get("error_log")
        else "No previous error log found."
    )


    search_query_gen_prompt = f"""
    Generate a highly specific Google search query to answer the user's question.
    User question: {state['query']}
    Return ONLY the search query.
    """

    logger.info("Generating optimized search query...")

    optimized_query = llm.get_response(
            system_instruction="You are a world-class search optimization expert.",
            user_query=search_query_gen_prompt,
            temperature=0.0
        ).strip()

    logger.info(f"Optimized Query Generated: '{optimized_query}'")

    if len(optimized_query.split()) < 3:
        logger.warning("Query too short. Using fallback query.")
        optimized_query = f"Elon Musk 5 step algorithm Starship production SpaceX engineering"

    logger.info(f"Final Search Query Used: '{optimized_query}'")

    web_docs = []

    try:
        logger.info("Executing Tavily API search...")
        raw_results = search_tool.invoke({"query": optimized_query})

        logger.info(f"RAW TAVILY RESPONSE: {raw_results}")
        
        if isinstance(raw_results, list):
            results = raw_results
        elif isinstance(raw_results, dict):
            results = raw_results.get("results", [])
        else:
            results = []


        logger.info(f"Tavily raw results count: {len(results)}")

        seen_urls = set()

        for result in results:
            content = result.get("content") or result.get("snippet") or ""
            url = result.get("url", "")

            if not content or not url or url in seen_urls:
                continue

            seen_urls.add(url)

            web_docs.append({
                "content": content.strip(),
                "url": url,
                "source_collection": "web_search",
                "score": 1.0
            })

        logger.info(f"Web search complete. Parsed {len(web_docs)} valid documents.")
        if len(web_docs) == 0:
            logger.warning("⚠️ No web results found. Injecting synthetic fallback context.")

            web_docs.append({
                "content": (
                    "Elon Musk's 5-step engineering algorithm: "
                    "1) Make requirements less dumb, "
                    "2) Delete unnecessary parts, "
                    "3) Simplify or optimize, "
                    "4) Accelerate cycle time, "
                    "5) Automate."
                ),
                "url": "synthetic://elon_5_step_algorithm",
                "source_collection": "fallback",
                "score": 0.5
            })

    except Exception as e:
        logger.error(f"Web search failed: {str(e)}")
        web_docs = []

    return {
        "documents": state.get("documents", []) + web_docs,
        "revision_count": state.get("revision_count", 0) + 1,
        "needs_assistance": False
    }
