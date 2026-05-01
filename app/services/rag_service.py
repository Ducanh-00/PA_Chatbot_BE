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
            from langchain_community.chat_models.tongyi import ChatTongyi
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain_qdrant import QdrantVectorStore
            
            # Khởi tạo LLM
            self.llm = ChatTongyi(
                model_name=settings.ALIBABA_CHAT_MODEL,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
            )
            
            # Sửa tham số model_name thành model
            self.embeddings = DashScopeEmbeddings(
                model=settings.ALIBABA_EMBEDDING_MODEL,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
            )
            
            # Khởi tạo Vector Store
            self.vector_store = QdrantVectorStore(
                client=qdrant_manager.get_client(),
                collection_name=settings.QDRANT_COLLECTION,
                embedding=self.embeddings,
            )

    def query(self, question: str):
        self._initialize()
        
        # Kiểm tra an toàn trước khi search
        if not self.vector_store:
            raise Exception("Vector store failed to initialize")
            
        docs = self.vector_store.similarity_search(question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        system_prompt = """Bạn là Petrolimex AI - Chuyên gia hỗ trợ kỹ thuật hàng không.
Trả lời dựa trên ngữ cảnh được cung cấp.

QUY TẮC ĐỊNH DẠNG:
1. Tiêu đề: Dùng ###.
2. In đậm: Dùng ** cho thông số kỹ thuật.
3. Danh sách: Dùng bullet points.
4. Emojis: Dùng ⛽, ✈️, 📋."""

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
