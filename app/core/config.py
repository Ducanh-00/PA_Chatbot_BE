import os
from dotenv import load_dotenv

load_dotenv()

def get_env_str(key: str, default: str = None) -> str:
    val = os.environ.get(key)
    return val.strip() if val is not None else default

class Settings:
    GOOGLE_API_KEY = get_env_str("GOOGLE_API_KEY")
    
    # Qdrant Config (Hỗ trợ cả Local và Cloud)
    QDRANT_URL = get_env_str("QDRANT_URL") # Ưu tiên dùng URL (cho Cloud)
    QDRANT_API_KEY = get_env_str("QDRANT_API_KEY") # Cần thiết cho Cloud
    
    QDRANT_HOST = get_env_str("QDRANT_HOST", "localhost")
    QDRANT_PORT = get_env_str("QDRANT_PORT", "6333")
    QDRANT_PATH = get_env_str("QDRANT_PATH", "./qdrant_db") # Dùng cho local mode nếu không có URL/Host
    
    COLLECTION_NAME = "pa_procedures"
    DASHSCOPE_API_KEY = get_env_str("DASHSCOPE_API_KEY")
    
    # Alibaba Cloud Models
    ALIBABA_EMBEDDING_MODEL = get_env_str("ALIBABA_EMBEDDING_MODEL", "text-embedding-v3")
    ALIBABA_VISION_MODEL = get_env_str("ALIBABA_VISION_MODEL", "qwen-vl-max")
    ALIBABA_CHAT_MODEL = get_env_str("ALIBABA_CHAT_MODEL", "qwen-plus")
    ALIBABA_COLLECTION_NAME = get_env_str("ALIBABA_COLLECTION_NAME", "pa_procedures_alibaba")

settings = Settings()
