import os
import dashscope
import json
from typing import List, Dict, Any
from app.core.config import settings

class RAGService:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self._initialized = False

    def _initialize(self):
        if self._initialized:
            return
            
        try:
            # QUAN TRỌNG: Thiết lập API Key và URL cho tài khoản Quốc tế
            dashscope.api_key = settings.DASHSCOPE_API_KEY
            dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
            
            from langchain_community.chat_models.tongyi import ChatTongyi
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain_qdrant import QdrantVectorStore
            
            # Khởi tạo LLM bản ổn định (qwen-plus)
            self.llm = ChatTongyi(
                model=settings.ALIBABA_CHAT_MODEL,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
                temperature=0.1,
                streaming=True
            )
            
            # Khởi tạo Embeddings
            self.embeddings = DashScopeEmbeddings(
                model=settings.ALIBABA_EMBEDDING_MODEL,
                dashscope_api_key=settings.DASHSCOPE_API_KEY
            )
            
            # Kết nối Vector Store (Qdrant)
            from qdrant_client import QdrantClient
            
            # Log thử cấu hình (không in API Key)
            print(f"Connecting to Qdrant at: {settings.QDRANT_URL or settings.QDRANT_HOST}")
            
            # Đảm bảo port là số nguyên, mặc định 6333 nếu lỗi
            try:
                q_port = int(settings.QDRANT_PORT) if settings.QDRANT_PORT else 6333
            except:
                q_port = 6333

            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                host=None if settings.QDRANT_URL else settings.QDRANT_HOST,
                port=None if settings.QDRANT_URL else q_port
            )
            
            self.vector_store = QdrantVectorStore(
                client=client,
                collection_name=settings.ALIBABA_COLLECTION_NAME,
                embedding=self.embeddings
            )
            
            self._initialized = True
            print(f"RAGService initialized successfully")
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"CRITICAL: RAGService initialization failed: {error_detail}")
            self.init_error = str(e)
            self._initialized = False

    def query(self, question: str):
        self._initialize()
        
        if not self.vector_store:
            raise Exception("Vector store failed to initialize")
            
        docs = self.vector_store.similarity_search(question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        system_prompt = """Bạn là Petrolimex AI - Chuyên gia hỗ trợ kỹ thuật hàng không chuyên nghiệp.
Nhiệm vụ của bạn là trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.

QUY TẮC ĐỊNH DẠNG CÂU TRẢ LỜI:
1. Cấu trúc rõ ràng: Sử dụng tiêu đề (###) cho các phần lớn.
2. Nhấn mạnh: Sử dụng **in đậm** cho các từ khóa hoặc lưu ý quan trọng.
3. Danh sách: Sử dụng bullet points rõ ràng.
4. Trực quan: Sử dụng các icon phù hợp như ⛽, ✈️, 📋, ⚠️.
5. Ngôn ngữ: Trả lời lịch sự, chính xác, ngắn gọn nhưng đầy đủ."""

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

    def stream_query(self, question: str):
        try:
            self._initialize()
        except Exception as e:
            yield {"type": "error", "data": f"Lỗi khởi tạo hệ thống: {str(e)}"}
            return
        
        if not self.vector_store:
            error_msg = getattr(self, 'init_error', 'Không xác định')
            yield {"type": "error", "data": f"Kết nối dữ liệu thất bại: {error_msg}"}
            return
            
        docs = self.vector_store.similarity_search(question, k=4)
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        context = "\n\n".join([doc.page_content for doc in docs])
        
        system_prompt = """Bạn là Petrolimex AI - Chuyên gia hỗ trợ kỹ thuật hàng không chuyên nghiệp.
Trả lời dựa trên ngữ cảnh cung cấp. Sử dụng Markdown đẹp mắt, icon chuyên nghiệp."""

        user_prompt = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 1. Trả về thông tin Nguồn ngay lập tức
        yield {"type": "sources", "data": sources}
        
        # 2. Stream text phản hồi (Instant - không có delay suy nghĩ)
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield {"type": "content", "data": chunk.content}

rag_service = RAGService()
