import os
import sys
import io
import re
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


# Fix Unicode cho terminal Windows một cách an toàn (tránh lỗi đè chữ)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "pa_procedures"


# Đã xóa phần Regex Keyword Extraction cũ vì thay thế bằng BM25 (Hybrid Search)


def deduplicate_docs(docs):
    """Loại bỏ document trùng lặp dựa trên nội dung, bất kể tên file khác nhau."""
    seen_content = set()
    unique = []
    for doc in docs:
        # Chuẩn hóa 400 ký tự đầu để so khớp (loại bỏ khoảng trắng, lowercase)
        # Việc này giúp loại bỏ các bản copy của cùng một nội dung trong các file khác nhau
        content_snippet = re.sub(r'\s+', '', doc.page_content[:400].lower())
        if content_snippet not in seen_content:
            seen_content.add(content_snippet)
            unique.append(doc)
    return unique

def reciprocal_rank_fusion(results_lists, k=60):
    """
    Thuật toán Reciprocal Rank Fusion (RRF) để gộp kết quả từ nhiều bộ tìm kiếm.
    Càng đứng top cao ở các bộ tìm kiếm, điểm RRF càng lớn.
    """
    fused_scores = {}
    for docs in results_lists:
        for rank, doc in enumerate(docs):
            doc_hash = doc.page_content
            if doc_hash not in fused_scores:
                fused_scores[doc_hash] = {"doc": doc, "score": 0.0}
            fused_scores[doc_hash]["score"] += 1.0 / (rank + 1 + k)
    
    # Sắp xếp theo score giảm dần
    reranked_results = [
        item["doc"]
        for item in sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    ]
    return reranked_results


# ==============================================================================
# PHẦN 3: MAIN
# ==============================================================================

def main():
    print("--- HỆ THỐNG RAG PETROLIMEX AVIATION (Ultra Search v5) ---")
    
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        client = QdrantClient(path=QDRANT_PATH)
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
        print(f"   ✅ Database loaded: {client.get_collection(COLLECTION_NAME).points_count} chunks.")
        
        # --- KHỞI TẠO HYBRID SEARCH (Ensemble Retriever) ---
        print("   ⏳ Đang khởi tạo bộ máy tìm kiếm BM25 (có thể mất vài giây để tải DB vào RAM)...")
        # Kéo toàn bộ văn bản từ Qdrant về để lập index BM25
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=50000,
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
                
        bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
        bm25_retriever.k = 15 # Lấy 15 kết quả tốt nhất từ BM25
        
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
        # KHÔNG DÙNG EnsembleRetriever của Langchain nữa vì phiên bản Langchain hiện tại bị thiếu module.
        # Chúng ta sẽ tự code thuật toán RRF (Reciprocal Rank Fusion) ở dưới.
        print("   ✅ Đã khởi tạo xong Hybrid Search (BM25 + Vector).")
        
    except Exception as e:
        print(f"   ❌ Lỗi DB: {e}")
        return
    
    try:
        # Tăng nhẹ temperature và thêm repeat_penalty để tránh vòng lặp (hallucination loop)
        llm = ChatOllama(model="qwen2.5", temperature=0.1, top_p=0.9, repeat_penalty=1.1)
        print("   ✅ LLM (Qwen 2.5) ready.")
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
                
            print("🔍 Đang tìm kiếm bằng Hybrid Search (BM25 + Vector)...")

            # Lấy kết quả từ BM25 (tìm từ khóa)
            bm25_docs = bm25_retriever.invoke(query)
            # Lấy kết quả từ Vector (tìm ngữ nghĩa)
            vector_docs = vector_retriever.invoke(query)
            
            # Gộp điểm bằng RRF
            all_docs = reciprocal_rank_fusion([bm25_docs, vector_docs])
            
            # Lọc các đoạn trùng lặp nội dung, giữ lại tối đa 15 đoạn cho AI
            docs = deduplicate_docs(all_docs)[:15]
            print(f"   📚 Đã tìm thấy {len(docs)} đoạn văn bản phù hợp nhất.")

            
            if not docs:
                print("   ⚠️ Không tìm thấy kết quả phù hợp.")
                continue

            # ─── BƯỚC 4: Prompt & LLM ───
            template = """Bạn là trợ lý AI chuyên tra cứu tài liệu nội bộ của Petrolimex Aviation (PA).

NGUYÊN TẮC BẮT BUỘC:
1. CHỈ trả lời dựa trên nội dung ngữ cảnh bên dưới. TUYỆT ĐỐI KHÔNG bịa hoặc suy đoán thông tin không có trong ngữ cảnh.
2. PHẢI trả lời hoàn toàn BẰNG TIẾNG VIỆT.
3. Trích dẫn nguyên văn từ tài liệu khi có thể, ghi rõ tên file nguồn.
4. Nếu ngữ cảnh không chứa đủ thông tin, hãy trả lời: "Tôi không tìm thấy thông tin này trong các tài liệu được cung cấp."
5. Trình bày có cấu trúc: dùng gạch đầu dòng, đánh số, in đậm khi cần thiết.
6. Trình bày thông tin đầy đủ, chi tiết nhưng TRÁNH việc lặp đi lặp lại cùng một nội dung nhiều lần.

Ngữ cảnh từ tài liệu:
{context}

Câu hỏi: {question}

Trả lời (Tiếng Việt, đầy đủ, chi tiết, có cấu trúc):"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # Format context rõ ràng hơn để LLM biết nguồn
            context_str = "\n\n---\n".join([
                f"[Nguồn: {d.metadata.get('source_file', 'N/A')} | Trang: {d.metadata.get('page', 'N/A')} | Thư mục: {d.metadata.get('category', 'N/A')}]\n{d.page_content}"
                for d in docs
            ])
            
            chain = (
                {"context": lambda x: context_str, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            response = chain.invoke(query)
            
            print("\n🤖 TRẢ LỜI:")
            print("-" * 60)
            print(response)
            print("-" * 60)
            
            # Hiển thị nguồn (loại trùng)
            print("\n📌 Nguồn tài liệu:")
            seen_sources = set()
            for d in docs[:8]:
                src = f"{d.metadata.get('source_file', 'N/A')} (Trang {d.metadata.get('page', 'N/A')})"
                if src not in seen_sources:
                    seen_sources.add(src)
                    print(f"   - {src}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
