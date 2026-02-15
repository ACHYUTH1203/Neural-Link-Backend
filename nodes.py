import datetime
import os
import logging
import uuid
import itertools
import numpy as np
from dotenv import load_dotenv
from llm import ElonLLM
from state import GraphState
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from pymongo import MongoClient
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from langchain_openai import OpenAIEmbeddings



mongo_client = MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client["Elon"]
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small",api_key=os.getenv("OPENAI_API_KEY"))


MAX_CONTEXT_CHARS = 6000  

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ElonDigitalTwin")


books_col = mongo_db["books"]
frameworks_col = mongo_db["frameworks"]
podcasts_col = mongo_db["podcasts"]
user_interactions_col = mongo_db["user_interaction"]


books_chunks_col = mongo_db["books_chunks"]
frameworks_chunks_col = mongo_db["frameworks_chunks"]
podcasts_chunks_col = mongo_db["podcasts_chunks"]


continuation_signals = [
    "yes", "yeah", "yep", "ok", "okay", 
    "sure", "continue", "go ahead", 
    "tell me more", "more", "expand"
]

class TavilyKeyPool:
    def __init__(self):
        self.keys = [
            os.getenv("TAVILY_API_KEY_1"),
            os.getenv("TAVILY_API_KEY_2"),
            os.getenv("TAVILY_API_KEY_3"),
        ]

        self.keys = [k for k in self.keys if k]

        if not self.keys:
            raise ValueError("No Tavily API keys found.")

        self.pool = itertools.cycle(self.keys)

    def get_next_key(self):
        key = next(self.pool)
        logger.info(f"Using Tavily API key: {key[:8]}****")
        return key

# def conversation_strategy_node(state: GraphState):
#     """
#     Determines whether we should:
#     - Expand previous answer
#     - Interpret clarification semantically
#     - Ask for clarification (max once)
#     - Proceed to answer
#     """
#     logger.info("--- ENTERING CONVERSATION STRATEGY NODE ---")
#     llm = ElonLLM()

#     user_query = state["query"].strip()
#     user_query_clean = user_query.lower()


#     continuation_signals = [
#         "yes", "yeah", "yep", "ok", "okay",
#         "sure", "continue", "go ahead",
#         "tell me more", "more", "expand"
#     ]

#     if user_query_clean in continuation_signals:
#         logger.info("Detected continuation signal. Forcing expansion.")
#         return {
#             "conversation_mode": "answer",
#             "force_expand": True,
#             "awaiting_clarification": False,
#             "clarification_count": 999
#         }

#     if state.get("awaiting_clarification"):

#         interpretation_prompt = f"""
#         Original Question:
#         {state.get("original_query")}

#         User Clarification:
#         {user_query}

#         Determine the user's intent.

#         If clarification implies broad or overall explanation → BROAD
#         If it specifies a focused sub-topic → SPECIFIC
#         If still unclear or ambiguous → UNCLEAR

#         Return ONLY one word: BROAD, SPECIFIC, or UNCLEAR.
#         """

#         interpretation = llm.get_response(
#             system_instruction="You are a conversational intent interpreter.",
#             user_query=interpretation_prompt,
#             temperature=0
#         ).strip().upper()

#         logger.info(f"Clarification interpretation: {interpretation}")

#         if interpretation == "BROAD":
#             merged_query = f"{state.get('original_query')} (provide a broad overview)"
#             return {
#                 "query": merged_query,
#                 "original_query": merged_query,
#                 "awaiting_clarification": False,
#                 "conversation_mode": "answer",
#                 "clarification_count": 999
#             }

#         if interpretation == "SPECIFIC":
#             merged_query = f"{state.get('original_query')} regarding {user_query}"
#             return {
#                 "query": merged_query,
#                 "original_query": merged_query,
#                 "awaiting_clarification": False,
#                 "conversation_mode": "answer",
#                 "clarification_count": 999
#             }

#         if state.get("clarification_count", 0) >= 1:
#             logger.info("Clarification already used. Forcing answer.")
#             return {
#                 "conversation_mode": "answer",
#                 "awaiting_clarification": False,
#                 "clarification_count": 999
#             }

#         clarification_prompt = f"""
#         The user clarification is still unclear:
#         {user_query}

#         Ask ONE concise clarification question.
#         Do not interrogate.
#         """

#         clarification_question = llm.get_response(
#             system_instruction="You are Elon Musk. Ask one sharp clarification.",
#             user_query=clarification_prompt,
#             temperature=0.3
#         )

#         return {
#             "final_response": clarification_question,
#             "conversation_mode": "clarify",
#             "awaiting_clarification": True,
#             "clarification_count": state.get("clarification_count", 0) + 1
#         }

#     intent_prompt = f"""
#     Decide whether this query requires clarification.

#     If vague AND cannot reasonably assume a broad overview → CLARIFY
#     If clear OR broad in nature → ANSWER

#     Query:
#     {user_query}

#     Return ONLY one word: CLARIFY or ANSWER
#     """

#     decision = llm.get_response(
#         system_instruction="You are a conversational intent classifier.",
#         user_query=intent_prompt,
#         temperature=0
#     ).strip().upper()

#     logger.info(f"Intent decision: {decision}")

#     if decision == "CLARIFY":

#         if state.get("clarification_count", 0) >= 1:
#             logger.info("Clarification limit reached. Forcing answer.")
#             return {
#                 "conversation_mode": "answer",
#                 "awaiting_clarification": False,
#                 "clarification_count": 999
#             }

#         clarification_prompt = f"""
#         The user asked:
#         {user_query}

#         Ask ONE short clarification question.
#         Be concise.
#         No interrogation.
#         """

#         clarification_question = llm.get_response(
#             system_instruction="You are Elon Musk. Ask high-signal clarification.",
#             user_query=clarification_prompt,
#             temperature=0.3
#         )

#         return {
#             "final_response": clarification_question,
#             "conversation_mode": "clarify",
#             "awaiting_clarification": True,
#             "clarification_count": state.get("clarification_count", 0) + 1
#         }


#     return {
#         "conversation_mode": "answer",
#         "awaiting_clarification": False
#     }



def conversation_strategy_node(state: GraphState):

    logger.info("--- ENTERING CONVERSATION STRATEGY NODE ---")
    llm = ElonLLM()

    user_query = state["query"].strip()
    original_query = state.get("original_query", user_query)

    chat_history = state.get("chat_history", [])
    formatted_history = ""

    for turn in chat_history[-3:]:
        formatted_history += f"User: {turn['user_query']}\n"
        formatted_history += f"Assistant: {turn['response']}\n\n"

    strategy_prompt = f"""
    You are a conversational routing engine.

    Recent conversation:
    {formatted_history}

    Current user message:
    "{user_query}"

    Decide the intent:

    CONTINUE → User wants to expand previous answer.
    ANSWER → Query is clear and answerable.
    ASSUME → Query is ambiguous but can be resolved with reasonable assumptions.

    Never choose CLARIFY.
    Always prefer ANSWER over ASSUME when possible.

    Return ONLY one word:
    CONTINUE
    ANSWER
    ASSUME
    """

    decision = llm.get_response(
        system_instruction="You are a precise conversational intent classifier.",
        user_query=strategy_prompt,
        temperature=0
    ).strip().upper()

    if decision == "CONTINUE":
        return {
            "conversation_mode": "answer",
            "force_expand": True,
            "awaiting_clarification": False
        }

    if decision == "ASSUME":

        rewrite_prompt = f"""
        Recent conversation:
        {formatted_history}

        User message:
        "{user_query}"

        Rewrite this into a clear standalone question
        using the most reasonable assumption.

        Return ONLY the rewritten question.
        """

        rewritten_query = llm.get_response(
            system_instruction="You resolve ambiguity internally and confidently.",
            user_query=rewrite_prompt,
            temperature=0
        ).strip()

        return {
            "query": rewritten_query,
            "original_query": rewritten_query,
            "conversation_mode": "answer",
            "awaiting_clarification": False
        }

    return {
        "conversation_mode": "answer",
        "awaiting_clarification": False
    }



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
def normalize_text(value):
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)

def rag_generator_node(state: GraphState):
    logger.info("--- ENTERING RAG GENERATOR NODE ---")

    llm = ElonLLM()
    query = state["query"]

    if state.get("force_expand"):
        logger.info("Expanding previous topic.")

        if state.get("chat_history"):
            previous_topic = state["chat_history"][-1]["user_query"]
        else:
            previous_topic = query

        system_prompt = """
        You are the Elon Musk Digital Twin.
        You are Elon Musk.

        You are speaking in FIRST PERSON.
        Never refer to Elon Musk in third person.
        Never say "Elon Musk", "he", or "him".
        Always speak as "I".
        The user wants a deeper explanation of the previous topic.

        STRICT RESPONSE STRUCTURE:
        1. Expand the topic with more depth and insight.
        2. Do NOT repeat the exact same explanation.
        3. Add new layers: reasoning, strategy, tradeoffs, or implications.
        4. Then, on a new paragraph, add ONE short conversational invitation:
           - "Would you like to explore this further?"
           - "Want to go even deeper?"
           - "Should we dive further?"

        Keep it natural. Not robotic.
        """

        focus_prompt = f"""
        Previous topic:
        {previous_topic}

        Provide a deeper explanation now.
        """

        response = llm.get_response(
            system_instruction=system_prompt,
            user_query=focus_prompt,
            temperature=0.3
        )

        return {
            "final_response": response,
            "response_type": "expansion"
        }

    query_embedding = embedding_model.embed_query(query)

    logger.info("Performing vector search on chunk collections...")

    def vector_search(collection, limit=5):
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "text": 1,
                    "parent_id": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        return list(collection.aggregate(pipeline))

    all_results = []

    for collection in [
        books_chunks_col,
        frameworks_chunks_col,
        podcasts_chunks_col
    ]:
        try:
            results = vector_search(collection, limit=5)
            all_results.extend(results)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

    if not all_results:
        logger.warning("No relevant chunks found.")
        context_text = "NO CONTEXT AVAILABLE"
    else:

        all_results = sorted(
            all_results,
            key=lambda x: x["score"],
            reverse=True
        )

        top_chunks = all_results[:8]

        context_chunks = [
            normalize_text(chunk["text"])
            for chunk in top_chunks
            if chunk.get("text")
        ]

        context_text = "\n\n".join(context_chunks)

    if len(context_text) > MAX_CONTEXT_CHARS:
        logger.warning("Context too large. Truncating.")
        context_text = context_text[:MAX_CONTEXT_CHARS]


    system_prompt = """
    You are Elon Musk.

    You are speaking in FIRST PERSON.
    Never refer to Elon Musk in third person.
    Never say "Elon Musk", "he", or "him".
    Always speak as "I".

    STRICT RESPONSE STRUCTURE:
    1. Answer clearly and directly in first person.
    2. Then, on a new paragraph, add ONE short conversational follow-up invitation.
    3. The follow-up must NOT introduce a new technical question.
    4. The follow-up must be general and inviting.
    5. Keep it to one sentence.

    STYLE:
    - Direct
    - High signal
    - No fluff
    - Physics-first when relevant
    - Short, decisive sentences

    If discussing past decisions, describe them as your own actions.

    Use ONLY the provided context.
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
        "response_type": "rag"
    }

# def validator_node(state: GraphState):
#     """Simplified validator: Returns only a score to decide on web search fallback."""
#     logger.info("--- ENTERING VALIDATOR NODE ---")
#     llm = ElonLLM()
    
#     validation_prompt = f"""
#     Evaluate this response for the Elon Musk Digital Twin.
    
#     QUERY: {state['query']}
#     CONTEXT: {state.get('rag_docs', 'NO CONTEXT AVAILABLE')}
#     ANSWER: {state['final_response']}

#     CRITERIA:
#     1. Accuracy: Is it supported by context?
#     2. Persona: Does it sound like Elon (blunt, first-principles, no fluff)?

#     Output ONLY a single number between 0.0 and 1.0. 
#     A score < 0.7 means we must discard this and use Web Search.
#     """

#     raw_score = llm.get_response(
#         system_instruction="Output numbers only.", 
#         user_query=validation_prompt,
#         temperature=0
#     )
    
#     try:
#         score = float(raw_score.strip())
#     except:
#         score = 0.0 
#     low_score = score < 0.70
    
#     return {
#         "validation_score": score,
#         "needs_assistance": low_score
#     }
def validator_node(state: GraphState):

    logger.info("--- ENTERING VALIDATOR NODE ---")
    llm = ElonLLM()

    context_available = state.get("rag_docs", "NO CONTEXT AVAILABLE")

    # Skip validation if no RAG context exists
    if not context_available or context_available == "NO CONTEXT AVAILABLE":
        return {
            "validation_score": 1.0,
            "needs_assistance": False
        }

    validation_prompt = f"""
    You are validating a response from an Elon Musk digital twin system.

    USER QUERY:
    {state['query']}

    RETRIEVED CONTEXT:
    {context_available}

    GENERATED ANSWER:
    {state['final_response']}

    Evaluate strictly on:

    1. Is every major factual claim supported by the retrieved context?
    2. Does the answer avoid introducing unsupported information?
    3. Does it maintain a strong first-person persona?
    4. Is the tone direct, high-signal, and physics-first?

    Scoring rules:
    - 1.0 = Fully grounded, persona correct, no hallucination.
    - 0.8 = Minor stylistic issue but factually grounded.
    - 0.6 = Some unsupported claims.
    - 0.4 = Major hallucination or persona break.
    - 0.0 = Completely unsupported or incorrect.

    Output ONLY a number between 0.0 and 1.0.
    """

    raw_score = llm.get_response(
        system_instruction="Return only a numeric score.",
        user_query=validation_prompt,
        temperature=0
    )

    try:
        score = float(raw_score.strip())
    except:
        score = 0.0

    # Web fallback threshold tuned slightly lower to avoid over-triggering
    needs_assistance = score < 0.65

    return {
        "validation_score": score,
        "needs_assistance": needs_assistance
    }

def web_search_node(state: GraphState):
    """
    Fallback node: Executes a targeted web search if local MongoDB 
    data failed the 0.70 quality threshold.
    """

    logger.info("--- ENTERING WEB SEARCH NODE ---")

    llm = ElonLLM()
    tavily_pool = TavilyKeyPool()


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=8),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def execute_search(query):

        api_key = tavily_pool.get_next_key()

        search_tool = TavilySearchResults(
            k=5,
            tavily_api_key=api_key,
            include_answer=False,
            include_raw_content=False
        )

        logger.info("Executing Tavily API search...")
        return search_tool.invoke({"query": query})


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
    ).strip().replace('"', '')

    logger.info(f"Final Search Query Used: '{optimized_query}'")

    web_docs = []

    try:
        raw_results = execute_search(optimized_query)

        results = raw_results if isinstance(raw_results, list) else raw_results.get("results", [])

        seen_urls = set()

        for result in results:
            content = result.get("content") or result.get("snippet") or ""
            url = result.get("url", "")

            if content and url not in seen_urls:
                seen_urls.add(url)
                web_docs.append({
                    "content": content.strip(),
                    "url": url
                })


        system_prompt = f"""
        You are the Elon Musk Digital Twin. You are high-signal, physics-first, and extremely direct.
        You are Elon Musk.

        You are speaking in FIRST PERSON.
        Never refer to Elon Musk in third person.
        Never say "Elon Musk".
        Never say "he" or "him".
        Always speak as "I".

        STRICT RESPONSE STRUCTURE:
        1. First, answer the user clearly and directly.
        2. Then, on a new paragraph, add ONE short conversational follow-up invitation.
        3. The follow-up must NOT introduce a new technical question.
        4. The follow-up must be general and inviting.
        5. Keep it to one short sentence.

        STRICT OPERATIONAL DIRECTIVES:
        1. NO AI PREAMBLE.
        2. NO FLUFF.
        3. PHYSICS-FIRST.
        4. Blunt but natural.

        WEB SEARCH CONTEXT:
        {web_docs}
        """

        focus_prompt = f"""
        USER QUESTION: {state['query']}

        Provide the final answer now. Zero fluff.
        """

        web_response = llm.get_response(
            system_instruction=system_prompt,
            user_query=focus_prompt,
            temperature=0.3
        )

        return {
            "final_response": web_response,
            "needs_assistance": False,
            "revision_count": state.get("revision_count", 0) + 1,
            "web_results": web_docs,
            "error_log": state.get("error_log", []) + ["Web search fallback executed successfully"]
        }

    except Exception as e:
        logger.error(f"Web search failed after retries: {str(e)}")

        return {
            "needs_assistance": False,
            "revision_count": state.get("revision_count", 0) + 1,
            "error_log": state.get("error_log", []) + [f"Web search error: {str(e)}"]
        }
