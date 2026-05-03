"""
Chat Service - Chứa toàn bộ business logic.
Không biết gì về HTTP request/response.
Chỉ làm việc với Repository và RAGService.
"""
from typing import AsyncGenerator

from app.repositories.chat_repository import ChatRepository
from app.services.rag_service import rag_service


class ChatService:
    def __init__(self, repo: ChatRepository):
        self.repo = repo

    async def create_session(self, first_message: str) -> dict:
        """Tạo session mới, lấy phần đầu câu hỏi làm tiêu đề."""
        title = first_message[:60].strip() or "New Chat"
        session = await self.repo.create_session(title)
        return {"id": session.id, "title": session.title, "created_at": session.created_at}

    async def list_sessions(self) -> list[dict]:
        """Lấy danh sách tất cả phiên chat."""
        sessions = await self.repo.get_all_sessions()
        return [
            {"id": s.id, "title": s.title, "created_at": s.created_at}
            for s in sessions
        ]

    async def get_session_detail(self, session_id: str) -> dict | None:
        """Lấy chi tiết 1 phiên chat kèm toàn bộ tin nhắn."""
        session = await self.repo.get_session_by_id(session_id)
        if not session:
            return None

        messages = await self.repo.get_messages_by_session(session_id)
        return {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at,
            "messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "sources": m.sources,
                    "created_at": m.created_at,
                }
                for m in messages
            ],
        }

    async def delete_session(self, session_id: str) -> bool:
        """Xóa phiên chat. Trả về False nếu không tìm thấy."""
        session = await self.repo.get_session_by_id(session_id)
        if not session:
            return False
        await self.repo.delete_session(session)
        return True

    async def save_user_message(self, session_id: str, content: str) -> None:
        """Lưu tin nhắn của user vào DB."""
        await self.repo.add_message(session_id=session_id, role="user", content=content)

    async def save_ai_message(
        self, session_id: str, content: str, sources: list
    ) -> None:
        """Lưu tin nhắn của AI vào DB."""
        await self.repo.add_message(
            session_id=session_id, role="assistant", content=content, sources=sources
        )

    async def stream_response(self, question: str) -> AsyncGenerator[dict, None]:
        """Stream câu trả lời từ RAG. Yield từng chunk."""
        for chunk in rag_service.stream_query(question):
            yield chunk
