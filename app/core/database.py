from qdrant_client import QdrantClient
from app.core.config import settings

class QdrantManager:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            # Ưu tiên kết nối qua URL (Qdrant Cloud hoặc Server Docker)
            if settings.QDRANT_URL:
                print(f"   [DATABASE] Connecting to Qdrant via URL: {settings.QDRANT_URL}")
                cls._client = QdrantClient(
                    url=settings.QDRANT_URL, 
                    api_key=settings.QDRANT_API_KEY
                )
            elif settings.QDRANT_HOST:
                # Dùng cho môi trường Docker Compose (host='qdrant')
                print(f"   [DATABASE] Connecting to Qdrant Host: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
                cls._client = QdrantClient(host=settings.QDRANT_HOST, port=int(settings.QDRANT_PORT))
            else:
                # Dùng cho chạy local file
                print(f"   [DATABASE] Initializing Local Qdrant Storage: {settings.QDRANT_PATH}")
                cls._client = QdrantClient(path=settings.QDRANT_PATH)
        return cls._client

qdrant_manager = QdrantManager()
