import os
import sys
import io
import re
import dashscope
from qdrant_client import QdrantClient
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from dotenv import load_dotenv

# Fix Unicode cho terminal Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import settings

# Load env và cấu hình DashScope International
load_dotenv()
dashscope.api_key = settings.DASHSCOPE_API_KEY
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = settings.ALIBABA_COLLECTION_NAME

def deduplicate_docs(docs):
    seen_content = set()
    unique = []
    for doc in docs:
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
    
    reranked_results = [
        item["doc"]
        for item in sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    ]
    return reranked_results

def main():
    print("--- HỆ THỐNG RAG PETROLIMEX AVIATION (ALIBABA CLOUD v1) ---")
    
    try:
        embeddings = DashScopeEmbeddings(
            model=settings.ALIBABA_EMBEDDING_MODEL,
            dashscope_api_key=settings.DASHSCOPE_API_KEY
        )
        client = QdrantClient(path=QDRANT_PATH)
        
        if not client.collection_exists(COLLECTION_NAME):
            print(f"   ❌ Collection '{COLLECTION_NAME}' không tồn tại. Vui lòng chạy script ingestion trước.")
            return

        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
        print(f"   ✅ Database loaded: {client.get_collection(COLLECTION_NAME).points_count} chunks.")
        
        print("   ⏳ Đang khởi tạo bộ máy tìm kiếm BM25...")
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        docs_for_bm25 = []
        for r in records:
            content = r.payload.get("page_content", "")
            if content:
                docs_for_bm25.append(Document(
                    page_content=content,
                    metadata=r.payload.get("metadata", {})
                ))
                
        if not docs_for_bm25:
            print("   ⚠️ Không có dữ liệu để khởi tạo BM25.")
            return

        bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
        bm25_retriever.k = 10
        
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        print("   ✅ Đã khởi tạo xong Hybrid Search (Alibaba).")
        
    except Exception as e:
        print(f"   ❌ Lỗi khởi tạo: {e}")
        return
    
    try:
        llm = ChatTongyi(
            model=settings.ALIBABA_CHAT_MODEL,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
            temperature=0.1
        )
        print(f"   ✅ LLM ({settings.ALIBABA_CHAT_MODEL}) ready.")
    except Exception as e:
        print(f"   ❌ Lỗi LLM: {e}")
        return

    while True:
        try:
            query = input("\n❓ Nhập câu hỏi: ")
            if query.lower() in ['exit', 'quit', 'thoát']:
                break
            if not query.strip():
                continue
                
            print("🔍 Đang tìm kiếm...")

            bm25_docs = bm25_retriever.invoke(query)
            vector_docs = vector_retriever.invoke(query)
            all_docs = reciprocal_rank_fusion([bm25_docs, vector_docs])
            docs = deduplicate_docs(all_docs)[:10]
            
            if not docs:
                print("   ⚠️ Không tìm thấy kết quả phù hợp.")
                continue

            print(f"   📚 Tìm thấy {len(docs)} đoạn văn bản. Đang tạo câu trả lời...")

            template = """Bạn là trợ lý AI chuyên tra cứu tài liệu nội bộ của Petrolimex Aviation (PA).

NGUYÊN TẮC BẮT BUỘC:
1. CHỈ trả lời dựa trên nội dung ngữ cảnh bên dưới. TUYỆT ĐỐI KHÔNG bịa hoặc suy đoán thông tin không có trong ngữ cảnh.
2. PHẢI trả lời hoàn toàn BẰNG TIẾNG VIỆT.
3. Trích dẫn nguyên văn từ tài liệu khi có thể, ghi rõ tên file nguồn.
4. Nếu ngữ cảnh không chứa đủ thông tin, hãy trả lời: "Tôi không tìm thấy thông tin này trong các tài liệu được cung cấp."
5. Trình bày có cấu trúc: dùng gạch đầu dòng, đánh số, in đậm khi cần thiết.

Ngữ cảnh từ tài liệu:
{context}

Câu hỏi: {question}

Trả lời (Tiếng Việt, đầy đủ, chi tiết, có cấu trúc):"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            context_str = "\n\n---\n".join([
                f"[Nguồn: {d.metadata.get('source_file', 'N/A')} | Trang: {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
                for d in docs
            ])
            
            chain = (
                {"context": lambda x: context_str, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            response = chain.invoke(query)
            
            print("\n🤖 TRẢ LỜI (ALIBABA):")
            print("-" * 60)
            print(response)
            print("-" * 60)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
