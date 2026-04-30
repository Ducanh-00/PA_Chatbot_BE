import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "pa_procedures"

def check_file_chunks(filename):
    print(f"🔍 Đang tìm các chunks của file: {filename} trong Qdrant DB...\n")
    
    try:
        client = QdrantClient(path=QDRANT_PATH)
        
        # Tạo filter để chỉ lấy các chunk thuộc về file này
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source_file",
                    match=models.MatchValue(value=filename)
                )
            ]
        )
        
        # Scroll để lấy tất cả kết quả phù hợp (giới hạn 100 chunk cho 1 file)
        records, next_page = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=query_filter,
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        if not records:
            print("❌ Không tìm thấy chunk nào cho file này! File chưa được nạp hoặc nạp thất bại.")
            return

        print(f"✅ Đã tìm thấy {len(records)} chunks. Dưới đây là nội dung chi tiết:\n")
        
        for i, record in enumerate(records):
            content = record.payload.get("page_content", "N/A")
            page = record.payload.get("metadata", {}).get("page", "N/A")
            extracted_by = record.payload.get("metadata", {}).get("extracted_by", "N/A")
            
            print(f"🌟 CHUNK #{i+1} [Trang {page} | Extracted by: {extracted_by}]")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(content)
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            
    except Exception as e:
        print(f"❌ Lỗi khi truy vấn Qdrant: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "01. XD Pham vi HT QL ATTT.docx"
        
    check_file_chunks(filename)
