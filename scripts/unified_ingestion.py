import os
import sys
import base64
import fitz
import pymupdf4llm
import time
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document
import uuid
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import settings

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data") 
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "pa_procedures"
PROCESSED_LOG_PATH = os.path.join(BASE_DIR, "processed_files.txt")

print("1. Init Local Embeddings (Ollama - nomic-embed-text)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(path=QDRANT_PATH)

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print("   Created new collection (Ollama 768d).")

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

print("2. Init Local Vision LLM (Ollama - LLaVA)...")
chat_model = ChatOllama(model="llava", temperature=0)

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

def generate_chunk_id(chunk: Document) -> str:
    content_to_hash = f"{chunk.metadata.get('source_file', '')}_{chunk.page_content}"
    hash_hex = hashlib.md5(content_to_hash.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hash_hex))

def process_unified_pdf(file_path, category, file_name):
    print(f"\n[UNIFIED MODE] Processing: {file_name}")
    all_chunks = []
    
    # 1. Trích xuất Text nhanh bằng pymupdf4llm
    try:
        md_pages_data = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        # Gộp text theo số trang để tránh lặp lại
        page_texts = {}
        for item in md_pages_data:
            p = item.get("metadata", {}).get("page", 1)
            page_texts.setdefault(p, []).append(item.get("text", ""))
    except Exception as e:
        print(f"  ❌ Error reading PDF text: {e}")
        return []

    # 2. Mở file bằng fitz để phân tích hình học
    doc = fitz.open(file_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Heuristics: Nhận diện trang phức tạp
        # Chỉ đánh dấu hình ảnh LỚN (sơ đồ/lưu đồ thật), bỏ qua logo/header nhỏ
        page_area = page.rect.width * page.rect.height
        has_large_images = False
        for img in page.get_images():
            xref = img[0]
            try:
                img_rect_list = page.get_image_rects(xref)
                for img_rect in img_rect_list:
                    img_area = img_rect.width * img_rect.height
                    if img_area > page_area * 0.15:  # Ảnh chiếm >15% trang = sơ đồ/lưu đồ thật
                        has_large_images = True
                        break
            except:
                pass
            if has_large_images:
                break
        
        tables = page.find_tables()
        has_tables = len(tables.tables) > 0 if hasattr(tables, 'tables') else False
        drawings = page.get_drawings()
        has_drawings = len(drawings) > 20  # Nếu có >20 nét vẽ thì khả năng cao là lưu đồ
        
        is_complex = has_large_images or has_drawings
        
        if is_complex:
            print(f"  -> Page {page_num + 1} is COMPLEX (LargeImg: {has_large_images}, Drawings: {has_drawings}, Tables: {has_tables} - ignored). Using LLaVA...")
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("jpeg")
            base64_img = base64.b64encode(img_bytes).decode('utf-8')
            
            messages = [
                HumanMessage(
                    content=[
                        {
                            "type": "text", 
                            "text": "You are a data extraction expert. Analyze the flowchart or diagram in the image. 1. Identify all decision nodes (diamonds/conditions). 2. List all process steps. 3. Describe the logical flow from Start to End in Markdown format. If it is a normal image, describe it in detail. Extract any relevant text clearly. Respond ONLY with the extracted Markdown content."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    ]
                )
            ]
            
            try:
                response = chat_model.invoke(messages)
                md_content = response.content
                extracted_by = "llava_vision"
            except Exception as e:
                print(f"     ❌ Error calling LLaVA on page {page_num+1}: {e}. Fallback to normal text.")
                md_content = "\n".join(page_texts.get(page_num + 1, [""]))
                extracted_by = "pymupdf4llm_fallback"
        else:
            print(f"  -> Page {page_num + 1} is SIMPLE. Using fast text extraction.")
            md_content = "\n".join(page_texts.get(page_num + 1, [""]))
            extracted_by = "pymupdf4llm"
            
        # Chia nhỏ đoạn text/markdown
        if md_content and md_content.strip():
            try:
                md_splits = markdown_splitter.split_text(md_content)
                chunks = text_splitter.split_documents(md_splits)
                
                for chunk in chunks:
                    chunk.metadata["category"] = category
                    chunk.metadata["source_file"] = file_name
                    chunk.metadata["page"] = page_num + 1
                    chunk.metadata["extracted_by"] = extracted_by
                    
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"     ❌ Error chunking page {page_num+1}: {e}")
            
    doc.close()
    return all_chunks

def load_processed_files():
    if not os.path.exists(PROCESSED_LOG_PATH):
        return set()
    with open(PROCESSED_LOG_PATH, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def mark_file_processed(file_path):
    with open(PROCESSED_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{file_path}\n")

if __name__ == "__main__":
    print("=======================================")
    print("UNIFIED INGESTION PIPELINE STARTING")
    print("=======================================")
    
    processed_files_set = load_processed_files()
    if processed_files_set:
        print(f"Loaded {len(processed_files_set)} already processed files. They will be skipped.")

    pdf_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith(".pdf") and not f.startswith("."):
                pdf_files.append(os.path.join(root, f))
                
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}/ to process.")
    else:
        for file_path in pdf_files:
            if file_path in processed_files_set:
                print(f"⏭️  Skipping already processed file: {os.path.basename(file_path)}")
                continue

            category = os.path.relpath(os.path.dirname(file_path), DATA_DIR)
            if category == '.':
                category = 'uncategorized'
                
            chunks = process_unified_pdf(file_path, category, os.path.basename(file_path))
            if chunks:
                print(f"  Total chunks extracted for {os.path.basename(file_path)}: {len(chunks)}")
                # Chia nho thanh batch 50 chunks de tranh Rate Limit
                BATCH_SIZE = 50
                saved = 0
                for i in range(0, len(chunks), BATCH_SIZE):
                    batch = chunks[i:i+BATCH_SIZE]
                    batch_ids = [generate_chunk_id(c) for c in batch]
                    for attempt in range(3):  # Thu lai toi da 3 lan
                        try:
                            vector_store.add_documents(documents=batch, ids=batch_ids)
                            saved += len(batch)
                            break
                        except Exception as e:
                            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                                wait = 60 * (attempt + 1)
                                print(f"     ⏳ Rate limit hit. Waiting {wait}s... (attempt {attempt+1}/3)")
                                time.sleep(wait)
                            else:
                                print(f"     ❌ Error saving batch: {e}")
                                break
                print(f"  ✅ Saved {saved}/{len(chunks)} chunks to Qdrant!")
            
            mark_file_processed(file_path)
