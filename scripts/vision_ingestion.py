import os
import sys
import base64
import fitz
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

print("1. Init Google Embeddings & Qdrant...")
embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
client = QdrantClient(path=QDRANT_PATH)

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )

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

def process_complex_pdf(file_path, category, file_name):
    print(f"\n[VISION MODE] Processing: {file_name}")
    doc = fitz.open(file_path)
    all_chunks = []
    
    for page_num, page in enumerate(doc):
        print(f"  -> Analyzing Page {page_num + 1}/{len(doc)} with LLaVA...")
        # Render trang PDF thành ảnh JPG
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("jpeg")
        base64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        # Gửi ảnh cho LLaVA
        messages = [
            HumanMessage(
                content=[
                    {
                        "type": "text", 
                        "text": "You are a data extraction expert. Look at this document page. If there is a flowchart, describe its logic step-by-step in extreme detail using Markdown. If there are tables, extract them exactly as Markdown tables. Extract all text clearly. Respond ONLY with the extracted Markdown content."
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
            
            # Chia nhỏ nội dung do LLaVA trả về
            md_splits = markdown_splitter.split_text(md_content)
            chunks = text_splitter.split_documents(md_splits)
            
            for chunk in chunks:
                chunk.metadata["category"] = category
                chunk.metadata["source_file"] = file_name
                chunk.metadata["page"] = page_num + 1
                chunk.metadata["extracted_by"] = "llava_vision"
                
            all_chunks.extend(chunks)
            print(f"     ✅ LLaVA extracted {len(chunks)} chunks from this page.")
        except Exception as e:
            print(f"     ❌ Error calling LLaVA on page {page_num+1}: {e}")
            
    doc.close()
    return all_chunks

DATA_VISION_DIR = os.path.join(BASE_DIR, "data_vision")
os.makedirs(DATA_VISION_DIR, exist_ok=True)

if __name__ == "__main__":
    print("WARNING: Running Vision Ingestion. It will take ~5-15s PER PAGE.")
    pdf_files = []
    
    for root, dirs, files in os.walk(DATA_VISION_DIR):
        for f in files:
            if f.lower().endswith(".pdf") and not f.startswith("."):
                pdf_files.append(os.path.join(root, f))
                
    if not pdf_files:
        print(f"No PDF files found in {DATA_VISION_DIR}/ to process.")
        print("Please place your complex PDFs (flowcharts, tables) into the 'data_vision' folder.")
    else:
        for file_path in pdf_files:
            category = os.path.relpath(os.path.dirname(file_path), DATA_VISION_DIR)
            if category == '.':
                category = 'uncategorized'
                
            chunks = process_complex_pdf(file_path, category, os.path.basename(file_path))
            if chunks:
                print(f"\nTotal chunks from Vision for {os.path.basename(file_path)}: {len(chunks)}")
                chunk_ids = [generate_chunk_id(c) for c in chunks]
                try:
                    vector_store.add_documents(documents=chunks, ids=chunk_ids)
                    print(f"✅ Successfully saved chunks for {os.path.basename(file_path)} to Qdrant!")
                except Exception as e:
                    print(f"❌ Error saving to Qdrant: {e}")
