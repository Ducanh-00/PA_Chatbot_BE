import os
import sys
import pickle
import hashlib
import uuid
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import settings
import dashscope
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document

def generate_chunk_id(chunk: Document) -> str:
    content_to_hash = f"{chunk.metadata.get('source_file', '')}_{chunk.page_content}"
    hash_hex = hashlib.md5(content_to_hash.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hash_hex))

def main():
    load_dotenv()
    dashscope.api_key = settings.DASHSCOPE_API_KEY
    dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

    COLLECTION_NAME = settings.ALIBABA_COLLECTION_NAME
    
    print(f"1. Loading saved chunks from all_chunks.pkl...")
    pkl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "all_chunks.pkl")
    
    if not os.path.exists(pkl_path):
        print(f"[ERROR] Cannot find {pkl_path}!")
        return

    with open(pkl_path, 'rb') as f:
        all_chunks = pickle.load(f)
        
    print(f"   -> Loaded {len(all_chunks)} chunks successfully!")

    print(f"2. Init Alibaba Embeddings ({settings.ALIBABA_EMBEDDING_MODEL})...")
    embeddings = DashScopeEmbeddings(
        model=settings.ALIBABA_EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY
    )

    print(f"3. Connecting to NEW Qdrant Cloud: {settings.QDRANT_URL}...")
    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=60.0  # Tăng thời gian chờ lên 60 giây để tránh lỗi timeout
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

    print(f"4. Pushing {len(all_chunks)} chunks to Qdrant Cloud...")
    
    # Push in batches of 200 (nhỏ hơn để giảm tải cho API và tránh timeout)
    batch_size = 200
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(all_chunks))
        batch_chunks = all_chunks[start_idx:end_idx]
        
        batch_ids = [generate_chunk_id(c) for c in batch_chunks]
        
        print(f"   Pushing batch {i+1}/{total_batches} (items {start_idx} to {end_idx})...")
        
        # Thêm cơ chế tự động thử lại (Retry) tối đa 3 lần nếu bị lỗi timeout
        max_retries = 3
        for attempt in range(max_retries):
            try:
                vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
                break # Nếu thành công thì thoát vòng lặp retry
            except Exception as e:
                print(f"   [ERROR] Failed to push batch {i+1} (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    print("   -> Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"   [CRITICAL] Batch {i+1} failed completely!")
            
    print("\n[FINISH] All data pushed to new Qdrant Cloud Cluster!")

if __name__ == "__main__":
    main()
