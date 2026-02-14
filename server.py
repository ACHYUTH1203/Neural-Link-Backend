
import uvicorn
import uuid
import datetime
import os
import random
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from pymongo import MongoClient
from graph import app  # LangGraph compiled app

from fastapi.middleware.cors import CORSMiddleware


mongo_client = MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client["Elon"]

usage_col = mongo_db["usage_tracking"]
waitlist_col = mongo_db["waitlist_emails"]


FREE_CHAT_LIMIT = 5
FREE_TOKEN_LIMIT = 8000


server = FastAPI(title="Elon Musk Digital Twin API")


server.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vercel.com/achyuths-projects-7ecd3fe6/neurallink/AUom3hXpgugsK8q9t8FWjSfcDhHz"],  #frontend domain 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str


class EmailRequest(BaseModel):
    email: str




def get_or_create_usage(user_id: str):
    usage = usage_col.find_one({"user_id": user_id})

    if not usage:
        usage = {
            "user_id": user_id,
            "chat_count": 0,
            "token_count": 0,
            "is_unlocked": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        usage_col.insert_one(usage)

    return usage

@server.get("/init")
async def init_chat(request: Request, response: Response):

    active_user_id = request.cookies.get("user_id")

    if not active_user_id:
        active_user_id = str(uuid.uuid4())
        response.set_cookie(
            key="user_id",
            value=active_user_id,
            httponly=True,
            max_age=60 * 60 * 24 * 30
        )

    usage = get_or_create_usage(active_user_id)

    remaining = max(0, FREE_CHAT_LIMIT - usage["chat_count"])

    intro_message = "Hey, I'm Elon’s digital twin. What do you want to explore?"



    return {
        "response": intro_message.strip(),
        "remaining_questions": remaining
    }


@server.post("/chat")
async def chat_endpoint(request: Request, chat_request: ChatRequest, response: Response):


    active_user_id = request.cookies.get("user_id")

    if not active_user_id:
        active_user_id = str(uuid.uuid4())
        response.set_cookie(
            key="user_id",
            value=active_user_id,
            httponly=True,
            max_age=60 * 60 * 24 * 30
        )

    usage = get_or_create_usage(active_user_id)

    if not usage.get("is_unlocked", False):

        if usage["chat_count"] >= FREE_CHAT_LIMIT:
            return {
                "limit_reached": True,
                "message": "Free limit reached. Enter your email to unlock full access."
            }

        if usage["token_count"] >= FREE_TOKEN_LIMIT:
            return {
                "limit_reached": True,
                "message": "Token limit reached. Enter your email to unlock full access."
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

        response_text = result.get("final_response", "")


        estimated_tokens = len(chat_request.query.split()) + len(response_text.split())

        usage_col.update_one(
            {"user_id": active_user_id},
            {
                "$inc": {
                    "chat_count": 1,
                    "token_count": estimated_tokens
                },
                "$set": {"updated_at": datetime.utcnow()}
            }
        )

        remaining = max(0, FREE_CHAT_LIMIT - (usage["chat_count"] + 1))

        return {
            "response": response_text,
            "remaining_questions": remaining,
            "limit_reached": False
        }

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail="Graph Execution Failed")
    
@server.post("/join-waitlist")
async def join_waitlist(request: Request, email_request: EmailRequest):

    active_user_id = request.cookies.get("user_id")

    if not active_user_id:
        raise HTTPException(status_code=400, detail="User session missing")

    # Prevent duplicate emails
    existing = waitlist_col.find_one({"email": email_request.email})

    if existing:
        return {
            "success": True,
            "message": "You're already on the waitlist."
        }

    waitlist_col.insert_one({
        "user_id": active_user_id,
        "email": email_request.email,
        "created_at": datetime.utcnow()
    })

    return {
        "success": True,
        "message": "You're on the waitlist. We'll keep you posted."
    }


@server.post("/unlock")
async def unlock_access(request: Request, email_request: EmailRequest):

    active_user_id = request.cookies.get("user_id")

    if not active_user_id:
        raise HTTPException(status_code=400, detail="User session missing")

    waitlist_col.update_one(
        {"email": email_request.email},
        {
            "$set": {
                "email": email_request.email,
                "user_id": active_user_id,
                "timestamp": datetime.utcnow()
            }
        },
        upsert=True
    )

    usage_col.update_one(
        {"user_id": active_user_id},
        {"$set": {"is_unlocked": True}}
    )

    return {
        "success": True,
        "message": "Access unlocked successfully."
    }


if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=8001)
