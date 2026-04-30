import os
import sys
import base64
import fitz
import pymupdf4llm
import time
import dashscope
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document
import uuid
import hashlib
import docx2txt
from dotenv import load_dotenv

# No stdout reconfig to avoid conflicts

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import settings

# Load env và cấu hình DashScope International
load_dotenv()
dashscope.api_key = settings.DASHSCOPE_API_KEY
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data") 
COLLECTION_NAME = settings.ALIBABA_COLLECTION_NAME
PROCESSED_LOG_PATH = os.path.join(BASE_DIR, "processed_files_alibaba.txt")

print(f"1. Init Alibaba Embeddings ({settings.ALIBABA_EMBEDDING_MODEL})...")
embeddings = DashScopeEmbeddings(
    model=settings.ALIBABA_EMBEDDING_MODEL,
    dashscope_api_key=settings.DASHSCOPE_API_KEY
)

# Kết nối lên Qdrant Cloud
print(f"2. Connecting to Qdrant Cloud: {settings.QDRANT_URL}...")
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print(f"   Created new collection '{COLLECTION_NAME}' on Cloud.")

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

print(f"3. Init Alibaba Vision LLM ({settings.ALIBABA_VISION_MODEL})...")
chat_model = ChatTongyi(
    model=settings.ALIBABA_VISION_MODEL,
    dashscope_api_key=settings.DASHSCOPE_API_KEY,
    temperature=0
)

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

def generate_chunk_id(chunk: Document) -> str:
    content_to_hash = f"{chunk.metadata.get('source_file', '')}_{chunk.page_content}"
    hash_hex = hashlib.md5(content_to_hash.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hash_hex))

def process_unified_pdf(file_path, category, file_name):
    print(f"\n[CLOUD INGESTION] Processing: {file_name}")
    all_chunks = []
    
    try:
        md_pages_data = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        page_texts = {}
        for item in md_pages_data:
            p = item.get("metadata", {}).get("page", 1)
            page_texts.setdefault(p, []).append(item.get("text", ""))
    except Exception as e:
        print(f"  [ERROR] Reading PDF text: {e}")
        return []

    doc = fitz.open(file_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_area = page.rect.width * page.rect.height
        has_large_images = False
        for img in page.get_images():
            xref = img[0]
            try:
                img_rect_list = page.get_image_rects(xref)
                for img_rect in img_rect_list:
                    img_area = img_rect.width * img_rect.height
                    if img_area > page_area * 0.15:
                        has_large_images = True
                        break
            except:
                pass
            if has_large_images:
                break
        
        tables = page.find_tables()
        has_tables = len(tables.tables) > 0 if hasattr(tables, 'tables') else False
        drawings = page.get_drawings()
        has_drawings = len(drawings) > 20
        
        is_complex = has_large_images or has_drawings
        
        if is_complex:
            print(f"  -> Page {page_num + 1} is COMPLEX. Using Vision LLM...")
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("jpeg")
            base64_img = base64.b64encode(img_bytes).decode('utf-8')
            
            messages = [
                HumanMessage(
                    content=[
                        {
                            "type": "text", 
                            "text": "Analyze the image and extract all text, tables, and flowcharts in Markdown format. Respond ONLY with Markdown."
                        },
                        {
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{base64_img}"
                        }
                    ]
                )
            ]
            
            try:
                response = chat_model.invoke(messages)
                md_content = response.content
                if isinstance(md_content, list):
                    md_content = "".join([item.get("text", "") for item in md_content if isinstance(item, dict)])
                extracted_by = "vision"
            except Exception as e:
                print(f"     [ERROR] Calling Vision: {e}. Fallback.")
                md_content = "\n".join(page_texts.get(page_num + 1, [""]))
                extracted_by = "fallback"
        else:
            md_content = "\n".join(page_texts.get(page_num + 1, [""]))
            extracted_by = "pymupdf4llm"
            
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
                print(f"     [ERROR] Chunking: {e}")
            
    doc.close()
    return all_chunks

def process_docx(file_path, category, file_name):
    print(f"\n[CLOUD INGESTION] Processing DOCX: {file_name}")
    try:
        text = docx2txt.process(file_path)
        if not text or not text.strip(): return []
        doc = Document(page_content=text, metadata={
            "source_file": file_name, "category": category, "page": 1, "extracted_by": "docx2txt"
        })
        return text_splitter.split_documents([doc])
    except Exception as e:
        print(f"  [ERROR]: {e}")
        return []

if __name__ == "__main__":
    if os.path.exists(PROCESSED_LOG_PATH): os.remove(PROCESSED_LOG_PATH)
    
    data_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if (f.lower().endswith(".pdf") or f.lower().endswith(".docx")) and not f.startswith("."):
                data_files.append(os.path.join(root, f))
                
    for file_path in data_files:
        file_name = os.path.basename(file_path)
        category = os.path.relpath(os.path.dirname(file_path), DATA_DIR)
        
        if file_path.lower().endswith(".pdf"):
            chunks = process_unified_pdf(file_path, category, file_name)
        else:
            chunks = process_docx(file_path, category, file_name)
        
        if chunks:
            print(f"  Uploading {len(chunks)} chunks to Qdrant Cloud...")
            batch_ids = [generate_chunk_id(c) for c in chunks]
            try:
                vector_store.add_documents(documents=chunks, ids=batch_ids)
                print(f"  [SUCCESS] Uploaded successfully!")
            except Exception as e:
                print(f"  [ERROR] Upload failed: {e}")
    
    print("\n[FINISH] All data pushed to Qdrant Cloud!")
