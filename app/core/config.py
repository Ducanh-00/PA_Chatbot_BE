import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    
    # Qdrant Config (Hỗ trợ cả Local và Cloud)
    QDRANT_URL = os.environ.get("QDRANT_URL") # Ưu tiên dùng URL (cho Cloud)
    QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") # Cần thiết cho Cloud
    
    QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
    QDRANT_PORT = os.environ.get("QDRANT_PORT", 6333)
    QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_db") # Dùng cho local mode nếu không có URL/Host
    
    COLLECTION_NAME = "pa_procedures"
    DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
    
    # Alibaba Cloud Models
    ALIBABA_EMBEDDING_MODEL = os.environ.get("ALIBABA_EMBEDDING_MODEL", "text-embedding-v3")
    ALIBABA_VISION_MODEL = os.environ.get("ALIBABA_VISION_MODEL", "qwen-vl-max")
    ALIBABA_CHAT_MODEL = os.environ.get("ALIBABA_CHAT_MODEL", "qwen-plus")
    ALIBABA_COLLECTION_NAME = os.environ.get("ALIBABA_COLLECTION_NAME", "pa_procedures_alibaba")

settings = Settings()
