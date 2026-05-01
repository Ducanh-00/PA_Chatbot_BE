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
            from langchain_alibaba import DashScopeChatModels
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain_qdrant import QdrantVectorStore
            
            self.llm = DashScopeChatModels(
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
        
        # 2. Xây dựng System Prompt "Đẹp" và chuyên nghiệp
        system_prompt = """Bạn là Petrolimex AI - Chuyên gia hỗ trợ kỹ thuật hàng không chuyên nghiệp.
Nhiệm vụ của bạn là trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.

QUY TẮC ĐỊNH DẠNG CÂU TRẢ LỜI:
1. Cấu trúc rõ ràng: Sử dụng tiêu đề (###) cho các phần lớn.
2. Nhấn mạnh: Sử dụng **in đậm** cho các thông số kỹ thuật, mã hiệu (với B12, B13...), hoặc các lưu ý quan trọng.
3. Danh sách: Sử dụng bullet points rõ ràng, có khoảng cách.
4. Chuyên nghiệp: Nếu có quy trình gồm nhiều bước, hãy đánh số 1, 2, 3...
5. Trực quan: Sử dụng các icon phù hợp như ⛽, ✈️, 📋, ⚠️ để làm nổi bật thông tin.
6. Ngôn ngữ: Trả lời lịch sự, chính xác, ngắn gọn nhưng đầy đủ.

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
