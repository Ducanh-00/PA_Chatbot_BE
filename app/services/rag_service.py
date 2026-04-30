import re
import os
import dashscope
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

from app.core.config import settings

def deduplicate_docs(docs):
    seen_content = set()
    unique = []
    for doc in docs:
        # Làm sạch nội dung để so sánh tránh trùng lặp gần đúng
        content_snippet = re.sub(r'\s+', '', doc.page_content[:400].lower())
        if content_snippet not in seen_content:
            seen_content.add(content_snippet)
            unique.append(doc)
    return unique

def reciprocal_rank_fusion(results_lists, k=60):
    fused_scores = {}
    for docs in results_lists:
        for rank, doc in enumerate(docs):
            doc_hash = doc.page_content
            if doc_hash not in fused_scores:
                fused_scores[doc_hash] = {"doc": doc, "score": 0.0}
            fused_scores[doc_hash]["score"] += 1.0 / (rank + 1 + k)
    
    reranked_results = []
    for item in sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True):
        doc = item["doc"]
        doc.metadata["relevance_score"] = item["score"]
        reranked_results.append(doc)
        
    return reranked_results

# Các Prompt cho Alibaba
REWRITE_PROMPT_STR = """Bạn là trợ lý AI nội bộ của Petrolimex Aviation. 
Nhiệm vụ của bạn là viết lại câu hỏi của người dùng để tối ưu hóa việc tìm kiếm trong cơ sở dữ liệu (Vector Search).
Hãy chuyển câu hỏi thành một câu truy vấn ngắn gọn, tập trung vào các từ khóa kỹ thuật chính liên quan đến quy trình hàng không.
Chỉ trả về câu hỏi đã viết lại, không giải thích gì thêm."""

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên tra cứu tài liệu nội bộ của Petrolimex Aviation (PA).

NGUYÊN TẮC BẮT BUỘC:
1. CHỈ trả lời dựa trên nội dung ngữ cảnh bên dưới. TUYỆT ĐỐI KHÔNG bịa hoặc suy đoán thông tin không có trong ngữ cảnh.
2. PHẢI trả lời hoàn toàn BẰNG TIẾNG VIỆT.
3. Trích dẫn nguyên văn từ tài liệu khi có thể, ghi rõ tên file nguồn và số trang.
4. Nếu ngữ cảnh không chứa đủ thông tin, hãy trả lời: "Tôi không tìm thấy thông tin này trong các tài liệu được cung cấp."
5. Trình bày có cấu trúc: dùng gạch đầu dòng, đánh số, in đậm khi cần thiết.
6. Trình bày thông tin đầy đủ, chi tiết. Với các bảng biểu và lưu đồ đã được trích xuất, hãy giải thích rõ ràng các bước và dữ liệu.

Ngữ cảnh từ tài liệu:
{context}

Câu hỏi: {question}

Trả lời (Tiếng Việt, đầy đủ, chi tiết, có cấu trúc):"""

class RAGService:
    def __init__(self):
        self.rag_chain = None
        self.rewrite_chain = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.llm = None
        
    def initialize(self):
        try:
            print("\n--- [AI INITIALIZATION START - ALIBABA CLOUD] ---")
            
            # Cấu hình DashScope
            dashscope.api_key = settings.DASHSCOPE_API_KEY
            dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
            
            print(f"1. Loading LLM (Alibaba: {settings.ALIBABA_CHAT_MODEL})...")
            self.llm = ChatTongyi(
                model=settings.ALIBABA_CHAT_MODEL,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
                temperature=0.1
            )
            
            print(f"2. Loading Embeddings (Alibaba: {settings.ALIBABA_EMBEDDING_MODEL})...")
            embeddings = DashScopeEmbeddings(
                model=settings.ALIBABA_EMBEDDING_MODEL,
                dashscope_api_key=settings.DASHSCOPE_API_KEY
            )
            
            from app.core.database import qdrant_manager
            client = qdrant_manager.get_client()
            
            collection_name = settings.ALIBABA_COLLECTION_NAME
            print(f"4. Loading Collection: {collection_name}...")
            
            if not client.collection_exists(collection_name):
                print(f"WARNING: Collection '{collection_name}' does not exist. Check your ingestion process.")
            
            vector_store = QdrantVectorStore(
                client=client, 
                collection_name=collection_name, 
                embedding=embeddings
            )
            self.vector_retriever = vector_store.as_retriever(search_kwargs={"k": 12})
            
            print("5. Building BM25 Index (Fetching docs from Qdrant)...")
            records, _ = client.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True
            )
            
            docs_for_bm25 = []
            for r in records:
                content = r.payload.get("page_content", "")
                if content:
                    docs_for_bm25.append(Document(
                        page_content=content,
                        metadata=r.payload.get("metadata", {})
                    ))
            
            if docs_for_bm25:
                self.bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
                self.bm25_retriever.k = 10
                print(f"DONE: BM25 Index built with {len(docs_for_bm25)} documents.")
            else:
                print("WARNING: BM25 index is empty.")

            print("6. Setting up LangChain Chains...")
            # Rewrite Chain
            rewrite_prompt = ChatPromptTemplate.from_messages([
                ("system", REWRITE_PROMPT_STR),
                ("human", "{question}")
            ])
            self.rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
            
            # RAG Chain
            rag_prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
            self.rag_chain = (
                {"context": lambda x: x["context"], "question": lambda x: x["question"]}
                | rag_prompt 
                | self.llm 
                | StrOutputParser()
            )
            
            print("SUCCESS: ALIBABA RAG SYSTEM READY!\n")
            
        except Exception as e:
            print(f"\nCRITICAL INITIALIZATION FAILURE: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    def rewrite_question(self, question: str) -> str:
        if not self.rewrite_chain: return question
        try:
            return self.rewrite_chain.invoke({"question": question})
        except:
            return question

    def search(self, query: str):
        if not self.vector_retriever:
            raise Exception("AI System chua duoc khoi tao")
            
        bm25_docs = []
        if self.bm25_retriever:
            bm25_docs = self.bm25_retriever.invoke(query)
            
        vector_docs = self.vector_retriever.invoke(query)
        all_docs = reciprocal_rank_fusion([bm25_docs, vector_docs])
        return deduplicate_docs(all_docs)[:10]

    def generate_answer(self, query: str, docs: list) -> str:
        if not self.rag_chain:
            raise Exception("AI System chua duoc khoi tao")
            
        context_str = "\n\n---\n".join([
            f"[Nguồn: {d.metadata.get('source_file', 'N/A')} | Trang: {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
            for d in docs
        ])
        
        return self.rag_chain.invoke({"context": context_str, "question": query})

rag_service = RAGService()
