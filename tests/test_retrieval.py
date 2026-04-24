import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Thêm đường dẫn gốc để import thư mục common
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from common.enums.messages import CLIMessages

# Config same as Ingestion
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "pa_procedures"
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"

print("Connecting to Qdrant DB...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
client = QdrantClient(path=QDRANT_PATH)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

print(CLIMessages.RETRIEVAL_WELCOME)

while True:
    query = input(CLIMessages.PROMPT_QUESTION)
    if query.lower() in ['exit', 'q', 'quit']:
        break
        
    print(CLIMessages.SEARCHING_RETRIEVAL)
    
    # Find top 3 most relevant text chunks
    results = vector_store.similarity_search(query, k=3)
    
    if not results:
        print(CLIMessages.NO_RESULTS)
    else:
        for i, doc in enumerate(results, 1):
            category = doc.metadata.get('category', 'Unknown')
            source = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            print(f"--- RESULT {i} ---")
            print(f"📁 Category: {category}")
            print(f"📄 Source: {source} (Page {page})")
            print(f"📝 Content:\n{doc.page_content[:500]}...\n")
    print("="*60 + "\n")
