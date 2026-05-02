"""
Chat Repository - Tầng duy nhất được phép truy vấn database.
Controller và Service không được gọi DB trực tiếp.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

from app.models.chat import ChatSession, ChatMessage


class ChatRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_session(self, title: str) -> ChatSession:
        session = ChatSession(title=title)
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get_all_sessions(self) -> list[ChatSession]:
        result = await self.db.execute(
            select(ChatSession).order_by(ChatSession.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_session_by_id(self, session_id: str) -> Optional[ChatSession]:
        result = await self.db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        return result.scalars().first()

    async def get_messages_by_session(self, session_id: str) -> list[ChatMessage]:
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        )
        return list(result.scalars().all())

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
        )
        self.db.add(message)
        await self.db.commit()
        return message

    async def delete_session(self, session: ChatSession) -> None:
        await self.db.delete(session)
        await self.db.commit()
