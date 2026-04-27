"""Script de xem nhanh noi dung trong Qdrant DB."""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import settings

from qdrant_client import QdrantClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "pa_procedures"

client = QdrantClient(path=QDRANT_PATH)

# 1. Tong so chunks trong DB
info = client.get_collection(COLLECTION_NAME)
total = info.points_count
print(f"=== QDRANT DB STATUS ===")
print(f"Total chunks in DB: {total}")
print(f"Collection: {COLLECTION_NAME}")
print()

# 2. Xem 5 chunks mau
if total > 0:
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )
    print(f"=== 5 SAMPLE CHUNKS ===")
    for i, point in enumerate(results[0]):
        payload = point.payload or {}
        metadata = payload.get("metadata", {})
        content = payload.get("page_content", "")[:200]
        print(f"\n--- Chunk {i+1} ---")
        print(f"  Source: {metadata.get('source_file', 'N/A')}")
        print(f"  Page: {metadata.get('page', 'N/A')}")
        print(f"  Category: {metadata.get('category', 'N/A')}")
        print(f"  Extracted by: {metadata.get('extracted_by', 'N/A')}")
        print(f"  Content preview: {content}...")

    # 3. Thong ke theo extracted_by
    print(f"\n=== STATS BY EXTRACTION METHOD ===")
    all_points = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=total,
        with_payload=True,
        with_vectors=False,
    )
    methods = {}
    files = set()
    for point in all_points[0]:
        payload = point.payload or {}
        metadata = payload.get("metadata", {})
        method = metadata.get("extracted_by", "unknown")
        methods[method] = methods.get(method, 0) + 1
        files.add(metadata.get("source_file", "unknown"))
    
    for method, count in sorted(methods.items()):
        print(f"  {method}: {count} chunks")
    print(f"\nTotal unique files processed: {len(files)}")

client.close()
