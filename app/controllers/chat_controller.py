from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_service import rag_service

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        answer = rag_service.query(request.message)
        return ChatResponse(answer=answer)
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    return {"status": "ok", "initialized": rag_service.is_initialized}
