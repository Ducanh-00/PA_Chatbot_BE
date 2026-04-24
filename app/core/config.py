import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    QDRANT_PATH = "./qdrant_db"
    COLLECTION_NAME = "pa_procedures"
    EMBEDDING_MODEL_NAME = "text-embedding-004"
    BM25_CACHE_PATH = "all_chunks.pkl"
    RERANKER_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
    LLM_MODEL = "gemini-1.5-flash"


settings = Settings()

if not settings.GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found. Set it via environment variables.")
