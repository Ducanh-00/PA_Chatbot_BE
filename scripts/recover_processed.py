import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from qdrant_client import QdrantClient

QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "pa_procedures"
DATA_DIR = os.path.join(BASE_DIR, "data") 
PROCESSED_LOG_PATH = os.path.join(BASE_DIR, "processed_files.txt")

def recover():
    if not os.path.exists(QDRANT_PATH):
        print("Qdrant DB not found.")
        return

    try:
        client = QdrantClient(path=QDRANT_PATH)
        if not client.collection_exists(COLLECTION_NAME):
            print("Collection does not exist.")
            return
            
        processed_basenames = set()
        next_page_offset = None
        print("Scanning Qdrant for processed files...")
        while True:
            records, next_page_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )
            for record in records:
                if record.payload and "metadata" in record.payload:
                    src = record.payload["metadata"].get("source_file")
                    if src:
                        processed_basenames.add(src)
            if next_page_offset is None:
                break
                
        print(f"Found {len(processed_basenames)} unique files in Qdrant.")
        
        pdf_files = []
        for root, dirs, files in os.walk(DATA_DIR):
            for f in files:
                if f.lower().endswith(".pdf") and not f.startswith("."):
                    pdf_files.append(os.path.join(root, f))
                    
        recovered_count = 0
        with open(PROCESSED_LOG_PATH, "w", encoding="utf-8") as f:
            for file_path in pdf_files:
                if os.path.basename(file_path) in processed_basenames:
                    f.write(f"{file_path}\n")
                    recovered_count += 1
                    
        print(f"✅ Successfully wrote {recovered_count} files to {PROCESSED_LOG_PATH}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    recover()
