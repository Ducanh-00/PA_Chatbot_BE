from app.services.rag_service import rag_service
from app.core.config import settings

print("Starting init...")
rag_service._initialize()
print(f"Initialized: {rag_service._initialized}")
if not rag_service._initialized:
    print(f"Init Error: {getattr(rag_service, 'init_error', 'None')}")
