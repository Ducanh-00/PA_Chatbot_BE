from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
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

@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest):
    try:
        def event_generator():
            try:
                for chunk in rag_service.stream_query(request.message):
                    # Trả về theo chuẩn Server-Sent Events (SSE)
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
                
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        print(f"Error in chat stream endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    # Chúng ta đã bỏ is_initialized trong code mới, nên cập nhật lại status
    return {"status": "ok", "backend": "online"}

