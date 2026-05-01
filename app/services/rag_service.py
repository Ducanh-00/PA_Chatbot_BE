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
            # QUAN TRỌNG: Thiết lập URL cho tài khoản Quốc tế
            dashscope.api_key = settings.DASHSCOPE_API_KEY
            dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
            
            from langchain_community.chat_models.tongyi import ChatTongyi
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain_qdrant import QdrantVectorStore
            
            # Khởi tạo LLM (Dùng 'model' thay cho 'model_name' để ổn định, bật streaming)
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
            
            # Khởi tạo Vector Store
            self.vector_store = QdrantVectorStore(
                client=qdrant_manager.get_client(),
                collection_name=settings.ALIBABA_COLLECTION_NAME,
                embedding=self.embeddings,
            )

    def query(self, question: str):
        self._initialize()
        
        if not self.vector_store:
            raise Exception("Vector store failed to initialize")
            
        docs = self.vector_store.similarity_search(question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # System Prompt để format câu trả lời đẹp
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
        
        # 1. Trả về thông tin Nguồn trước tiên
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        yield {"type": "sources", "data": sources}
        
        # 2. Trả về từng đoạn text khi AI đang sinh (streaming)
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield {"type": "content", "data": chunk.content}

rag_service = RAGService()
