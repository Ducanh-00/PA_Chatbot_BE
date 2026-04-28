import os
import sys
import fitz
import pymupdf4llm
import uuid
import hashlib
import json
import time
from docx import Document as DocxDocument

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data") 
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "pa_procedures"
STAGED_IMAGES_DIR = os.path.join(BASE_DIR, "data", "staged_images")
STEP1_LOG_PATH = os.path.join(BASE_DIR, "step1_processed_files.txt")

os.makedirs(STAGED_IMAGES_DIR, exist_ok=True)

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

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

def generate_chunk_id(chunk: Document) -> str:
    content_to_hash = f"{chunk.metadata.get('source_file', '')}_{chunk.page_content}"
    hash_hex = hashlib.md5(content_to_hash.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hash_hex))

def load_step1_processed():
    if os.path.exists(STEP1_LOG_PATH):
        with open(STEP1_LOG_PATH, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def mark_step1_processed(file_path):
    with open(STEP1_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")

def process_docx(file_path, category, file_name):
    print(f"  -> Processing DOCX: {file_name}")
    try:
        doc = DocxDocument(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        
        md_content = "\n\n".join(full_text)
        all_chunks = []
        if md_content.strip():
            md_splits = markdown_splitter.split_text(md_content)
            chunks = text_splitter.split_documents(md_splits)
            for chunk in chunks:
                chunk.metadata["category"] = category
                chunk.metadata["source_file"] = file_name
                chunk.metadata["page"] = 1 # Docx doesn't have reliable page numbers
                chunk.metadata["extracted_by"] = "python-docx"
            all_chunks.extend(chunks)
        return all_chunks, False # No complex vision needed for docx usually
    except Exception as e:
        print(f"     ❌ Error processing docx: {e}")
        return [], False

def process_cpu_extraction(file_path, category, file_name):
    # Handle DOCX
    if file_path.lower().endswith(".docx"):
        return process_docx(file_path, category, file_name)
        
    # Handle PDF
    print(f"  -> Analyzing PDF Layout: {file_name}")
    doc = fitz.open(file_path)
    page_texts = pymupdf4llm.to_markdown(file_path, page_chunks=True)
    
    all_chunks = []
    has_complex = False
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Check for images or complex drawings
        images = page.get_images(full=True)
        has_large_images = False
        if images:
            page_area = page.rect.width * page.rect.height
            for img in images:
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    img_area = pix.width * pix.height
                    if img_area > page_area * 0.15:
                        has_large_images = True
                        break
                except: pass
        
        drawings = page.get_drawings()
        has_drawings = len(drawings) > 20
        is_complex = has_large_images or has_drawings
        
        if is_complex:
            has_complex = True
            pix = page.get_pixmap(dpi=150)
            unique_id = hashlib.md5(f"{file_name}_{page_num}".encode("utf-8")).hexdigest()
            img_path = os.path.join(STAGED_IMAGES_DIR, f"{unique_id}.jpg")
            json_path = os.path.join(STAGED_IMAGES_DIR, f"{unique_id}.json")
            pix.save(img_path)
            
            metadata = {
                "source_file": file_name,
                "category": category,
                "page": page_num + 1,
                "fallback_text": page_texts[page_num]["text"] if page_num < len(page_texts) else ""
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False)
        else:
            md_content = page_texts[page_num]["text"] if page_num < len(page_texts) else ""
            if md_content.strip():
                md_splits = markdown_splitter.split_text(md_content)
                chunks = text_splitter.split_documents(md_splits)
                for chunk in chunks:
                    chunk.metadata.update({"category": category, "source_file": file_name, "page": page_num + 1, "extracted_by": "pymupdf4llm"})
                all_chunks.extend(chunks)
                
    doc.close()
    return all_chunks, has_complex

if __name__ == "__main__":
    print("=======================================")
    print("STEP 1: CPU EXTRACTION PIPELINE (PDF + DOCX)")
    print("=======================================")
    
    processed_files_set = load_step1_processed()
    files_to_process = []
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if (f.lower().endswith(".pdf") or f.lower().endswith(".docx")) and not f.startswith("."):
                full_path = os.path.join(root, f)
                if full_path not in processed_files_set:
                    files_to_process.append(full_path)
                
    if not files_to_process:
        print("No new files found to process.")
    else:
        print(f"Found {len(files_to_process)} new files to process.")
        for file_path in files_to_process:
            file_name = os.path.basename(file_path)
            category = os.path.relpath(os.path.dirname(file_path), DATA_DIR)
            
            chunks, has_complex = process_cpu_extraction(file_path, category, file_name)
            
            if chunks:
                BATCH_SIZE = 50
                for i in range(0, len(chunks), BATCH_SIZE):
                    batch = chunks[i:i+BATCH_SIZE]
                    batch_ids = [generate_chunk_id(c) for c in batch]
                    try:
                        vector_store.add_documents(documents=batch, ids=batch_ids)
                    except Exception as e:
                        print(f"     ❌ Save error: {e}")
                print(f"  ✅ Saved {len(chunks)} chunks for {file_name}")
            
            mark_step1_processed(file_path)

    print("\n✅ Step 1 Complete!")
