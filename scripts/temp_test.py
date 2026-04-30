import os
import sys
import io
import unicodedata

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def normalize_text(text):
    if not text:
        return ""
    return unicodedata.normalize('NFC', text)

from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")

client = QdrantClient(path=QDRANT_PATH)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = QdrantVectorStore(client=client, collection_name="pa_procedures", embedding=embeddings)

query = "ISMS được áp dụng đối với các quy trình, tài sản thông tin, hệ thống thông tin, con người và cơ sở hạ tầng"
results = vector_store.similarity_search_with_score(query, k=15)

for r in results:
    file_name = normalize_text(r[0].metadata.get('source_file'))
    print(f"Score: {r[1]:.4f} - File: {file_name}")
