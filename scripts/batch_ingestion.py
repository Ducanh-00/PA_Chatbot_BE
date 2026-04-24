from docx import Document
import os
import pickle
import time
import pymupdf4llm
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from app.core.config import settings

# Config - Use absolute paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
CHUNKS_PATH = os.path.join(BASE_DIR, "all_chunks.pkl")
PROGRESS_PATH = os.path.join(BASE_DIR, "ingestion_progress.txt")
COLLECTION_NAME = "pa_procedures"

# 1. Init Embedding Model (Google API)
print(f"Loading Google embedding model: {settings.EMBEDDING_MODEL_NAME}...")
embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME)

# 2. Init Qdrant (Local)
print("Initializing Qdrant Database...")
client = QdrantClient(path=QDRANT_PATH)

# Chỉ tạo collection mới nếu chưa tồn tại (KHÔNG xóa dữ liệu cũ để hỗ trợ resume)
if not client.collection_exists(COLLECTION_NAME):
    print(f"Creating new collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
else:
    count = client.get_collection(COLLECTION_NAME).points_count
    print(
        f"Collection '{COLLECTION_NAME}' already exists with {count} points. Resuming..."
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# Config Splitter (Markdown -> Char)
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", ""],
    length_function=len,
)

print("Initializing PyMuPDF4LLM Engine (Super Fast Markdown Extraction)...")


def process_documents():
    all_chunks = []

    # Scan all pdf and docx in subdirectories
    for root, dirs, files in os.walk(DATA_DIR):
        # Get the name of the first-level directory as category
        relative_path = os.path.relpath(root, DATA_DIR)
        if relative_path == ".":
            continue

        category = relative_path.split(os.sep)[0]

        for file in files:
            file_path = os.path.join(root, file)
            # Skip hidden macOS files like .DS_Store
            if file.startswith("."):
                continue

            # Only process PDF and DOCX
            if not file.lower().endswith((".pdf", ".docx", ".doc")):
                continue

            # Bỏ qua file .doc nếu đã có bản .docx (đã convert)
            if file.lower().endswith(".doc") and not file.lower().endswith(".docx"):
                docx_version = file_path + "x"
                if os.path.exists(docx_version):
                    print(f"⏭️  Skipping {file} (đã có bản .docx)")
                    continue

            print(f"Processing (PyMuPDF4LLM): {file_path} (Category: {category})")

            try:
                if file.lower().endswith(".pdf"):
                    # Bước 1: Thử trích xuất text bình thường
                    md_content = pymupdf4llm.to_markdown(file_path)

                    # Bước 2: Nếu nội dung trống → PDF scan → dùng OCR
                    clean_text = md_content.strip().replace("\n", "").replace(" ", "")
                    if len(clean_text) < 50:
                        print("  ⚠️ PDF scan detected! Switching to OCR (Tesseract)...")
                        import fitz

                        doc = fitz.open(file_path)
                        ocr_pages = []
                        for page_num, page in enumerate(doc):
                            try:
                                tp = page.get_textpage_ocr(language="vie+eng", dpi=300)
                                page_text = page.get_text(textpage=tp)
                                if page_text.strip():
                                    ocr_pages.append(
                                        f"## Page {page_num + 1}\n{page_text}"
                                    )
                            except Exception as ocr_err:
                                print(
                                    f"    OCR error on page {page_num + 1}: {ocr_err}"
                                )
                        doc.close()
                        md_content = "\n\n".join(ocr_pages)
                        if md_content.strip():
                            print(
                                f"  ✅ OCR extracted {len(ocr_pages)} pages successfully!"
                            )
                        else:
                            print("  ❌ OCR could not extract any text.")
                else:
                    # Fallback to standard loader for DOCX
                    loader = Docx2txtLoader(file_path)
                    doc_objs = loader.load()
                    md_content = "\n".join([d.page_content for d in doc_objs])

                # Split documents by Markdown header tags
                md_splits = markdown_splitter.split_text(md_content)

                # Additional split by character limit (avoid overly long tables)
                chunks = text_splitter.split_documents(md_splits)

                # Add metadata tags for each document
                for chunk in chunks:
                    chunk.metadata["category"] = category
                    chunk.metadata["source_file"] = file

                all_chunks.extend(chunks)
                print(f"  -> Generated {len(chunks)} chunks.")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return all_chunks


# Tạo UUID duy nhất cho từng chunk dựa trên nội dung
def generate_chunk_id(chunk: Document) -> str:
    import hashlib
    import uuid

    # Hash nội dung + source file để đảm bảo duy nhất
    content_to_hash = f"{chunk.metadata.get('source_file', '')}_{chunk.page_content}"
    hash_hex = hashlib.md5(content_to_hash.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hash_hex))


# 4. Execute
chunks = process_documents()

if chunks:
    print(f"\nTotal chunks generated: {len(chunks)}")

    # 4.1 Tạo ID cho tất cả chunks
    chunk_ids = [generate_chunk_id(chunk) for chunk in chunks]

    # 4.2 Lấy danh sách ID đã có trong Qdrant
    print("Checking for existing chunks in Qdrant to avoid duplicate API calls...")
    existing_ids = set()
    # Scroll qua tất cả các record trong Qdrant để lấy ID
    offset = None
    while True:
        try:
            records, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=False,
                with_vectors=False,
                offset=offset,
            )
            for record in records:
                existing_ids.add(str(record.id))
            if offset is None:
                break
        except Exception as e:
            print(f"Collection might be empty or error reading: {e}")
            break

    print(f"Found {len(existing_ids)} existing chunks in database.")

    # 4.3 Lọc ra các chunks MỚI cần nạp
    new_chunks = []
    new_chunk_ids = []
    for chunk, cid in zip(chunks, chunk_ids):
        if cid not in existing_ids:
            new_chunks.append(chunk)
            new_chunk_ids.append(cid)

    print(
        f"🔥 Found {len(new_chunks)} NEW chunks to process (Skipping {len(chunks) - len(new_chunks)} existing chunks)"
    )

    if len(new_chunks) > 0:
        print("Pushing NEW data to Qdrant Vector Database...")
        import time

        batch_size = 20  # Giảm xuống 20 để tránh vượt quota

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            batch_ids = new_chunk_ids[i : i + batch_size]

            success = False
            retry_count = 0
            while not success and retry_count < 10:
                try:
                    # Truyền thêm tham số ids vào để ghi đè đúng ID
                    vector_store.add_documents(documents=batch, ids=batch_ids)
                    success = True
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        wait_time = min((retry_count + 1) * 15, 120)
                        print(
                            f"\n⚠️ Rate limit! Chờ {wait_time}s... (Thử lại {retry_count + 1}/10)"
                        )
                        time.sleep(wait_time)
                        retry_count += 1
                    else:
                        print(f"\n❌ Lỗi: {e}")
                        raise e

            if not success:
                print(f"\n❌ Dừng tại batch {i}. Chạy lại script để tiếp tục.")
                break

            print(f"  -> Inserted {i + len(batch)} / {len(new_chunks)} NEW chunks")
            time.sleep(3)
    else:
        print("✅ No new chunks to insert. Database is up to date!")

    print("\nSaving all chunks to local file for BM25 Keyword Search...")
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ All documents have been successfully ingested into Qdrant & Local Cache!")
else:
    print("No valid documents found to process.")
