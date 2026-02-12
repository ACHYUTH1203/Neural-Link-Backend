import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from graph import app  # Your compiled LangGraph instance

server = FastAPI(title="Elon Musk Digital Twin API")

class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default_session"

@server.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. Thread configuration for memory persistence
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # 2. Critical: Initialize the state with empty defaults
    # This prevents KeyErrors in nodes like rag_generator_node
    initial_state = {
        "query": request.query,
        "documents": [],
        "error_log": [],
        "revision_count": 0,
        "needs_assistance": False
    }

    try:
        # 3. Use ainvoke for async compatibility with FastAPI
        result = await app.ainvoke(initial_state, config=config)
        
        return {
            "response": result.get("final_response"),
            "score": result.get("validation_score"),
            "needs_web_search": result.get("needs_assistance"),
            "logs": result.get("error_log", [])
        }
    except Exception as e:
        # This will print the actual error to your terminal for debugging
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph Execution Failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=8001)