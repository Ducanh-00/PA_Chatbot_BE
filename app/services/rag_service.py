import dashscope
from app.core.config import settings
from app.core.database import qdrant_manager
import numpy as np

class RAGService:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        
    def _initialize(self):
        """Lazy initialization of AI components"""
        if self.llm is None:
            # Sửa tên Class đúng của langchain-alibaba
            from langchain_alibaba import ChatDashScope
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain_qdrant import QdrantVectorStore
            
            self.llm = ChatDashScope(
                model_name=settings.ALIBABA_CHAT_MODEL,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
            )
            self.embeddings = DashScopeEmbeddings(
                model_name=settings.ALIBABA_EMBEDDING_MODEL,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
            )
            self.vector_store = QdrantVectorStore(
                client=qdrant_manager.get_client(),
                collection_name=settings.QDRANT_COLLECTION,
                embedding=self.embeddings,
            )

    def query(self, question: str):
        self._initialize()
        
        # 1. Tìm kiếm tài liệu liên quan
        docs = self.vector_store.similarity_search(question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 2. System Prompt "Đẹp" và chuyên nghiệp
        system_prompt = """Bạn là Petrolimex AI - Chuyên gia hỗ trợ kỹ thuật hàng không chuyên nghiệp.
Nhiệm vụ của bạn là trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.

QUY TẮC ĐỊNH DẠNG CÂU TRẢ LỜI:
1. Cấu trúc rõ ràng: Sử dụng tiêu đề (###) cho các phần lớn.
2. Nhấn mạnh: Sử dụng **in đậm** cho các thông số kỹ thuật hoặc các lưu ý quan trọng.
3. Danh sách: Sử dụng bullet points rõ ràng.
4. Chuyên nghiệp: Sử dụng các icon phù hợp như ⛽, ✈️, 📋, ⚠️.
5. Ngôn ngữ: Trả lời lịch sự, chính xác, ngắn gọn nhưng đầy đủ.

Nếu thông tin không có trong ngữ cảnh, hãy lịch sự thông báo và trả lời dựa trên kiến thức chuyên môn của bạn nhưng phải có ghi chú rõ ràng."""

        user_prompt = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            "answer": response.content,
            "sources": [doc.metadata.get("source", "Unknown") for doc in docs]
        }

rag_service = RAGService()
