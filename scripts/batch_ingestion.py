import os
import glob
import pickle
import pymupdf4llm
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
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
COLLECTION_NAME = "pa_procedures"
# 1. Init Embedding Model (Google API)
print(f"Loading Google embedding model: {settings.EMBEDDING_MODEL_NAME}...")
embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME)

# 2. Init Qdrant (Local)
print("Initializing Qdrant Database...")
client = QdrantClient(path=QDRANT_PATH)

# Delete old collection if exists to reload
if client.collection_exists(COLLECTION_NAME):
    print(f"Delete old collection '{COLLECTION_NAME}'...")
    client.delete_collection(collection_name=COLLECTION_NAME)

# Create new collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
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

            print(f"Processing (PyMuPDF4LLM): {file_path} (Category: {category})")

            try:
                if file.lower().endswith(".pdf"):
                    # Convert PDF to Markdown (Preserves Tables magically fast)
                    md_content = pymupdf4llm.to_markdown(file_path)
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


# 4. Execute
chunks = process_documents()

if chunks:
    print(f"\nTotal chunks generated: {len(chunks)}")
    print("Pushing data to Qdrant Vector Database... (This might take a while)")

    # Batch size to avoid RAM overflow
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vector_store.add_documents(batch)
        print(f"  -> Inserted {i + len(batch)} / {len(chunks)} chunks")

    print("Saving all chunks to local file for BM25 Keyword Search...")
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ All documents have been successfully ingested into Qdrant & Local Cache!")
else:
    print("No valid documents found to process.")
