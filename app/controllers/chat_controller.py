"""
Chat Session Controller - Chỉ xử lý HTTP layer.
Mọi logic đều được ủy quyền cho ChatService.
"""
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.repositories.chat_repository import ChatRepository
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


def get_chat_service(db: AsyncSession = Depends(get_db)) -> ChatService:
    """Dependency injection: tạo Service với Repository từ DB session."""
    repo = ChatRepository(db)
    return ChatService(repo)


# ---------- Schemas ----------

class StreamRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


# ---------- Endpoints ----------

@router.post("/stream")
async def stream_chat(
    request: StreamRequest,
    service: ChatService = Depends(get_chat_service),
):
    """Stream câu trả lời RAG và lưu tin nhắn vào DB."""

    # Lưu tin nhắn user ngay lập tức (nếu có session)
    if request.session_id:
        await service.save_user_message(request.session_id, request.message)

    async def event_generator():
        full_response = ""
        sources_data = []
        try:
            async for chunk in service.stream_response(request.message):
                if chunk["type"] == "content":
                    full_response += chunk["data"]
                elif chunk["type"] == "sources":
                    sources_data = chunk["data"]
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # Lưu câu trả lời AI sau khi stream hoàn tất
            if request.session_id and full_response:
                await service.save_ai_message(
                    request.session_id, full_response, sources_data
                )
        except Exception as e:
            error_payload = {"type": "error", "data": str(e)}
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/status")
async def get_status():
    return {"status": "ok", "backend": "online"}
