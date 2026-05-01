from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.rag_service import rag_service

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = []

@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Nhận kết quả là một Dictionary từ rag_service
        result = rag_service.query(request.message)
        
        # Bóc tách và truyền vào ChatResponse đúng chuẩn
        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", [])
        )
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    # Chúng ta đã bỏ is_initialized trong code mới, nên cập nhật lại status
    return {"status": "ok", "backend": "online"}
