# import uvicorn
# import uuid
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Optional
# from graph import app  # Your compiled LangGraph instance

# server = FastAPI(title="Elon Musk Digital Twin API")

# class ChatRequest(BaseModel):
#     query: str
#     user_id: Optional[str] = None  # User ID for MongoDB & persistence

# @server.post("/chat")
# async def chat_endpoint(request: ChatRequest):
#     # 1. Identity Management: Use provided ID or generate a new one
#     # This ID links the local LangGraph memory and the MongoDB user_interaction collection
#     active_user_id = request.user_id if request.user_id else str(uuid.uuid4())
    
#     # 2. Thread configuration for LangGraph's InMemorySaver
#     config = {"configurable": {"thread_id": active_user_id}}
    
#     # 3. Initialize State
#     # original_query: preserves the user's raw input before refinement
#     # user_id: required by nodes.py to fetch/save MongoDB history
#     initial_state = {
#         "query": request.query,
#         "original_query": request.query,
#         "user_id": active_user_id,
#         "documents": [],
#         "error_log": [],
#         "revision_count": 0,
#         "needs_assistance": False
#     }

#     try:
#         # 4. Execute the Graph
#         result = await app.ainvoke(initial_state, config=config)
        
#         # 5. Return response along with the user_id for the frontend to store
#         return {
#             "user_id": active_user_id,
#             "refined_query": result.get("query"), # Useful for debugging context
#             "response": result.get("final_response"),
#             "score": result.get("validation_score"),
#             "needs_web_search": result.get("needs_assistance"),
#             "logs": result.get("error_log", [])
#         }
        
#     except Exception as e:
#         print(f"CRITICAL ERROR: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Graph Execution Failed: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run(server, host="0.0.0.0", port=8001)




import uvicorn
import uuid
import os
import random  
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from typing import Optional
from pymongo import MongoClient
from graph import app  

mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
mongo_db = mongo_client["Elon"]

user_interactions_col = mongo_db["user_interaction"]
waitlist_col = mongo_db["waitlist_emails"]

FREE_LIMIT = 5


server = FastAPI(title="Elon Musk Digital Twin API")


class ChatRequest(BaseModel):
    query: str


class EmailRequest(BaseModel):
    email: str


# @server.post("/chat")
# async def chat_endpoint(request: Request, chat_request: ChatRequest, response: Response):

#     active_user_id = request.cookies.get("user_id")

#     if not active_user_id:
#         active_user_id = str(uuid.uuid4())
#         response.set_cookie(
#             key="user_id",
#             value=active_user_id,
#             httponly=True,      
#             max_age=60*60*24*30 
#         )

#     question_count = user_interactions_col.count_documents(
#         {"user_id": active_user_id}
#     )

#     if question_count >= FREE_LIMIT:
#         return {
#             "limit_reached": True,
#             "message": "Free limit reached. Enter your email to unlock full access."
#         }

#     config = {"configurable": {"thread_id": active_user_id}}


#     initial_state = {
#     "query": chat_request.query,
#     "original_query": chat_request.query,
#     "user_id": active_user_id,
#     "documents": [],
#     "error_log": [],
#     "revision_count": 0,
#     "needs_assistance": False,
#     "awaiting_clarification": False,
#     "clarification_count": 0,
# }


#     try:
#         result = await app.ainvoke(initial_state, config=config)

#         return {
#             "response": result.get("final_response"),
#             "remaining_questions": FREE_LIMIT - question_count - 1
#         }

#     except Exception as e:
#         print(f"CRITICAL ERROR: {str(e)}")
#         raise HTTPException(status_code=500, detail="Graph Execution Failed")


# @server.post("/join-waitlist")
# async def join_waitlist(request: Request, email_request: EmailRequest):

#     active_user_id = request.cookies.get("user_id")

#     if not active_user_id:
#         raise HTTPException(status_code=400, detail="User session missing")

#     waitlist_col.insert_one({
#         "user_id": active_user_id,
#         "email": email_request.email,
#         "timestamp": datetime.utcnow()
#     })

#     return {"message": "You're on the waitlist."}

# @server.post("/chat")
# async def chat_endpoint(request: Request, chat_request: ChatRequest, response: Response):

#     active_user_id = request.cookies.get("user_id")

#     if not active_user_id:
#         active_user_id = str(uuid.uuid4())
#         response.set_cookie(
#             key="user_id",
#             value=active_user_id,
#             httponly=True,
#             max_age=60*60*24*30
#         )

#     question_count = user_interactions_col.count_documents(
#         {"user_id": active_user_id}
#     )

#     # --------------------------------------------
#     # ✅ BOT INITIATES CONVERSATION (FIRST LOAD)
#     # --------------------------------------------
#     # if question_count == 0 and chat_request.query.strip().lower() in ["", "hi", "hello", "hey"]:
#     #     return {
#     #         "response": "Welcome. What do you want to explore today — rockets, AI, manufacturing, or first principles?",
#     #         "remaining_questions": FREE_LIMIT
#     #     }
#     if question_count == 0 and chat_request.query.strip() == "":
    
#         greetings = [
#             "Welcome. What would you like to explore today?",
#             "Good to see you. What’s on your mind?",
#             "Let’s think in first principles. What topic shall we break down?"
#         ]

#         return {
#             "response": random.choice(greetings),
#             "remaining_questions": FREE_LIMIT
#         }

#     # --------------------------------------------
#     # FREE LIMIT CHECK
#     # --------------------------------------------
#     if question_count >= FREE_LIMIT:
#         return {
#             "limit_reached": True,
#             "message": "Free limit reached. Enter your email to unlock full access."
#         }

#     config = {"configurable": {"thread_id": active_user_id}}

#     initial_state = {
#         "query": chat_request.query,
#         "original_query": chat_request.query,
#         "user_id": active_user_id,
#         "documents": [],
#         "error_log": [],
#         "revision_count": 0,
#         "needs_assistance": False,
#         "awaiting_clarification": False,
#         "clarification_count": 0,
#     }

#     try:
#         result = await app.ainvoke(initial_state, config=config)

#         return {
#             "response": result.get("final_response"),
#             "remaining_questions": FREE_LIMIT - question_count - 1
#         }

#     except Exception as e:
#         print(f"CRITICAL ERROR: {str(e)}")
#         raise HTTPException(status_code=500, detail="Graph Execution Failed")


# if __name__ == "__main__":
#     uvicorn.run(server, host="0.0.0.0", port=8001)



import uvicorn
import uuid
import os
import random
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from typing import Optional
from pymongo import MongoClient
from graph import app  

mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
mongo_db = mongo_client["Elon"]

user_interactions_col = mongo_db["user_interaction"]
waitlist_col = mongo_db["waitlist_emails"]

FREE_LIMIT = 5



import uvicorn
import uuid
import os
import random
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from typing import Optional
from pymongo import MongoClient
from graph import app  

mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
mongo_db = mongo_client["Elon"]

user_interactions_col = mongo_db["user_interaction"]
waitlist_col = mongo_db["waitlist_emails"]

FREE_LIMIT = 5

server = FastAPI(title="Elon Musk Digital Twin API")


class ChatRequest(BaseModel):
    query: str


class EmailRequest(BaseModel):
    email: str


# =====================================================
# ✅ NEW INIT ENDPOINT (BOT STARTS CONVERSATION)
# =====================================================
@server.get("/init")
async def init_chat(request: Request, response: Response):

    active_user_id = request.cookies.get("user_id")

    if not active_user_id:
        active_user_id = str(uuid.uuid4())
        response.set_cookie(
            key="user_id",
            value=active_user_id,
            httponly=True,
            max_age=60*60*24*30
        )

    greetings = [
        "Welcome. What would you like to explore today?",
        "Good to see you. What’s on your mind?",
        "Let’s think in first principles. What topic shall we break down?"
    ]

    return {
        "response": random.choice(greetings),
        "remaining_questions": FREE_LIMIT
    }


# =====================================================
# MAIN CHAT ENDPOINT
# =====================================================
@server.post("/chat")
async def chat_endpoint(request: Request, chat_request: ChatRequest, response: Response):

    active_user_id = request.cookies.get("user_id")

    if not active_user_id:
        active_user_id = str(uuid.uuid4())
        response.set_cookie(
            key="user_id",
            value=active_user_id,
            httponly=True,
            max_age=60*60*24*30
        )

    question_count = user_interactions_col.count_documents(
        {"user_id": active_user_id}
    )

    # --------------------------------------------
    # FREE LIMIT CHECK
    # --------------------------------------------
    if question_count >= FREE_LIMIT:
        return {
            "limit_reached": True,
            "message": "Free limit reached. Enter your email to unlock full access."
        }

    config = {"configurable": {"thread_id": active_user_id}}

    initial_state = {
        "query": chat_request.query,
        "original_query": chat_request.query,
        "user_id": active_user_id,
        "documents": [],
        "error_log": [],
        "revision_count": 0,
        "needs_assistance": False,
        "awaiting_clarification": False,
        "clarification_count": 0,
    }

    try:
        result = await app.ainvoke(initial_state, config=config)

        return {
            "response": result.get("final_response"),
            "remaining_questions": FREE_LIMIT - question_count - 1
        }

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail="Graph Execution Failed")


if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=8001)

app = FastAPI(title="Elon Musk Digital Twin API")


class ChatRequest(BaseModel):
    query: str


class EmailRequest(BaseModel):
    email: str


# =====================================================
# ✅ NEW INIT ENDPOINT (BOT STARTS CONVERSATION)
# =====================================================
@app.get("/init")
async def init_chat(request: Request, response: Response):

    active_user_id = request.cookies.get("user_id")

    if not active_user_id:
        active_user_id = str(uuid.uuid4())
        response.set_cookie(
            key="user_id",
            value=active_user_id,
            httponly=True,
            max_age=60*60*24*30
        )

    greetings = [
        "Welcome. What would you like to explore today?",
        "Good to see you. What’s on your mind?",
        "Let’s think in first principles. What topic shall we break down?"
    ]

    return {
        "response": random.choice(greetings),
        "remaining_questions": FREE_LIMIT
    }


# =====================================================
# MAIN CHAT ENDPOINT
# =====================================================
@app.post("/chat")
async def chat_endpoint(request: Request, chat_request: ChatRequest, response: Response):

    active_user_id = request.cookies.get("user_id")

    if not active_user_id:
        active_user_id = str(uuid.uuid4())
        response.set_cookie(
            key="user_id",
            value=active_user_id,
            httponly=True,
            max_age=60*60*24*30
        )

    question_count = user_interactions_col.count_documents(
        {"user_id": active_user_id}
    )

    # --------------------------------------------
    # FREE LIMIT CHECK
    # --------------------------------------------
    if question_count >= FREE_LIMIT:
        return {
            "limit_reached": True,
            "message": "Free limit reached. Enter your email to unlock full access."
        }

    config = {"configurable": {"thread_id": active_user_id}}

    initial_state = {
        "query": chat_request.query,
        "original_query": chat_request.query,
        "user_id": active_user_id,
        "documents": [],
        "error_log": [],
        "revision_count": 0,
        "needs_assistance": False,
        "awaiting_clarification": False,
        "clarification_count": 0,
    }

    try:
        result = await app.ainvoke(initial_state, config=config)

        return {
            "response": result.get("final_response"),
            "remaining_questions": FREE_LIMIT - question_count - 1
        }

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail="Graph Execution Failed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

