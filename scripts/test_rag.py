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
from qdrant_client.http import models

# Fix Unicode cho terminal Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "pa_procedures"


# ==============================================================================
# PHẦN 1: TRÍCH XUẤT KEYWORD TỪ CÂU HỎI
# ==============================================================================

def extract_file_reference(query):
    """Trích xuất tham chiếu file/tài liệu từ câu hỏi (nếu có).
    
    Chỉ kích hoạt khi người dùng RÕ RÀNG nhắc đến tên file/tài liệu.
    Trả về list keyword để filter, hoặc [] nếu không tìm thấy.
    """
    keywords = []
    
    # Pattern: Người dùng nhắc rõ "file/tài liệu + tên"
    # Ví dụ: "file 01. XD Pham vi", "tài liệu QT.AT.01", "quy trình CNTT.01"
    doc_words = r'(?:file|tài liệu|tl|document|văn bản|quy trình|quy định)'
    # Các từ báo hiệu kết thúc tên file, bắt đầu phần câu hỏi
    stop_words = (
        r'(?:,|\?|thì\b|là gì|là\b|có\b|nói\b|đề cập|quy định\b|mục đích|nội dung'
        r'|phạm vi|gồm|bao gồm|ghi\b|viết\b|trình bày|hướng dẫn|cho biết'
        r'|như thế nào|ra sao|ở đâu|bao nhiêu|khi nào|tại sao|để làm gì)'
    )
    
    file_ref_patterns = [
        # "nội dung của file/tài liệu + tên" (đặt trước vì cụ thể hơn)
        rf'(?:nội dung|nd)\s+(?:của\s+)?{doc_words}\s+["\']?(.+?)["\']?\s*{stop_words}',
        rf'(?:nội dung|nd)\s+(?:của\s+)?{doc_words}\s+["\']?(.+?)["\']?\s*$',
        # "trong file X, ..." hoặc "file X là gì"
        rf'(?:trong\s+)?{doc_words}\s+["\']?(.+?)["\']?\s*{stop_words}',
        # "file X" ở cuối câu
        rf'(?:trong\s+)?{doc_words}\s+["\']?(.+?)["\']?\s*(?:\?|$)',
    ]
    
    # Danh sách các từ phổ biến không nên coi là tên file
    common_verbs = {'xác định', 'tìm', 'xem', 'hỏi', 'cho', 'biết', 'lấy', 'đọc'}
    
    for pattern in file_ref_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            file_ref = match.group(1).strip().rstrip(',. ?')
            # Dọn dẹp: loại bỏ các từ hỏi bị dính ở cuối
            file_ref = re.sub(r'\s+(?:là gì\??|có gì|nói gì|gồm gì|gồm những gì)\s*$', '', file_ref, flags=re.IGNORECASE)
            
            # Kiểm tra nếu file_ref không phải là động từ phổ biến và đủ dài
            if len(file_ref) >= 3 and file_ref.lower() not in common_verbs:
                keywords.append(file_ref)
                # Nếu tên dài, tách thêm các phần có nghĩa
                # "01. XD Pham vi HT QL ATTT" -> thêm "XD Pham vi", "ATTT"
                parts = re.split(r'[\d]+[.\s]+', file_ref, maxsplit=1)
                if len(parts) > 1 and len(parts[1].strip()) >= 3:
                    keywords.append(parts[1].strip())
            break
    
    # Pattern: Mã tài liệu chuyên biệt (QT.XX.01, QD.XX.01, BM.XX, etc.)
    code_match = re.findall(r'((?:QT|QD|QĐ|BM|ISMS|SMS)[.\s-][A-Za-z]+[.\s-]?\d*)', query, re.IGNORECASE)
    keywords.extend(code_match)
    
    # Deduplicate
    seen = set()
    unique = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen and len(kw) >= 3:
            seen.add(kw_lower)
            unique.append(kw)
    
    return unique


# ==============================================================================
# PHẦN 2: TÌM KIẾM THÔNG MINH
# ==============================================================================

def search_by_file_filter(vector_store, query, keywords, k=10):
    """Tìm kiếm semantic nhưng filter theo tên file/category.
    
    Dùng should (OR) để match cả source_file VÀ category.
    """
    all_filtered = []
    
    for kw in keywords:
        try:
            file_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.source_file",
                        match=models.MatchText(text=kw)
                    ),
                    models.FieldCondition(
                        key="metadata.category",
                        match=models.MatchText(text=kw)
                    ),
                ]
            )
            docs = vector_store.as_retriever(
                search_kwargs={"k": k, "filter": file_filter}
            ).invoke(query)
            all_filtered.extend(docs)
        except Exception:
            pass
    
    return all_filtered


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
    except Exception as e:
        print(f"   ❌ Lỗi DB: {e}")
        return
    
    try:
        llm = ChatOllama(model="llama3", temperature=0)
        print("   ✅ LLM ready.")
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

            # ─── BƯỚC 1: Semantic Search (Tăng k để lấy nhiều dữ liệu hơn) ───
            semantic_docs = vector_store.as_retriever(
                search_kwargs={"k": 30}
            ).invoke(query)
            print(f"   📚 Semantic search: {len(semantic_docs)} kết quả")

            # ─── BƯỚC 2: File Filter Search (CHỈ KHI người dùng nhắc đến file) ───
            file_keywords = extract_file_reference(query)
            filtered_docs = []
            if file_keywords:
                print(f"   🎯 Phát hiện tham chiếu tài liệu: {file_keywords}")
                filtered_docs = search_by_file_filter(
                    vector_store, query, file_keywords, k=20
                )
                print(f"   📁 File filter: {len(filtered_docs)} kết quả")

            # ─── BƯỚC 3: Merge & Deduplicate ───
            # Lấy nhiều kết quả rồi lọc trùng, sau đó giữ lại tối đa 20 đoạn để LLM đọc
            all_docs = filtered_docs + semantic_docs
            docs = deduplicate_docs(all_docs)[:20]
            
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
6. KHÔNG ĐƯỢC tóm tắt quá ngắn gọn. Hãy trình bày cực kỳ CHI TIẾT và ĐẦY ĐỦ nhất mọi thông tin tìm thấy trong ngữ cảnh.

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
