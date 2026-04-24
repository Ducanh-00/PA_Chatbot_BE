import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    QDRANT_PATH = "./qdrant_db"
    COLLECTION_NAME = "pa_procedures"
    EMBEDDING_MODEL_NAME = "models/gemini-embedding-2"
    LLM_MODEL = "models/gemini-2.0-flash"


settings = Settings()

if not settings.GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found. Set it via environment variables.")
