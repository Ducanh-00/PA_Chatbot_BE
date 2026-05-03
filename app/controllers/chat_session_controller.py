"""
Chat Session Controller - Chỉ xử lý HTTP layer cho CRUD phiên chat.
Mọi logic đều được ủy quyền cho ChatService.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.repositories.chat_repository import ChatRepository
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/v1/chat/sessions", tags=["Chat Sessions"])


def get_chat_service(db: AsyncSession = Depends(get_db)) -> ChatService:
    return ChatService(ChatRepository(db))


# ---------- Schemas ----------

class CreateSessionRequest(BaseModel):
    title: str


# ---------- Endpoints ----------

@router.post("")
async def create_session(
    request: CreateSessionRequest,
    service: ChatService = Depends(get_chat_service),
):
    return await service.create_session(request.title)


@router.get("")
async def list_sessions(service: ChatService = Depends(get_chat_service)):
    return await service.list_sessions()


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
):
    detail = await service.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Session not found")
    return detail


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
):
    success = await service.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success"}
