from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from pydantic import BaseModel
from datetime import datetime

from app.core.database import get_db
from app.models.chat import ChatSession, ChatMessage

router = APIRouter(prefix="/api/v1/chat/sessions", tags=["Chat Sessions"])

class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources: list | dict | None = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class ChatSessionDetailResponse(ChatSessionResponse):
    messages: List[ChatMessageResponse]

class CreateSessionRequest(BaseModel):
    title: str

@router.post("", response_model=ChatSessionResponse)
async def create_session(request: CreateSessionRequest, db: AsyncSession = Depends(get_db)):
    new_session = ChatSession(title=request.title)
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)
    return new_session

@router.get("", response_model=List[ChatSessionResponse])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ChatSession).order_by(ChatSession.created_at.desc()))
    return result.scalars().all()

@router.get("/{session_id}", response_model=ChatSessionDetailResponse)
async def get_session_detail(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    session = result.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    msg_result = await db.execute(select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()))
    messages = msg_result.scalars().all()
    
    return ChatSessionDetailResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        messages=messages
    )

@router.delete("/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    session = result.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    await db.delete(session)
    await db.commit()
    return {"status": "success", "message": "Session deleted"}
