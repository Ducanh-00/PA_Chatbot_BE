import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Thêm đường dẫn gốc của project vào sys.path để có thể chạy python app/main.py trực tiếp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag_service import rag_service
from app.services.alibaba_rag_service import alibaba_rag_service
from app.controllers.chat_controller import router as chat_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*50)
    print("[STARTUP] INITIALIZING AI CORES...")
    print("="*50)
    
    # Init Local RAG
    print("\n--- Initializing Local AI Core ---")
    try:
        rag_service.initialize()
    except Exception as e:
        print(f"Local AI Init failed: {e}")

    # Init Alibaba RAG
    print("\n--- Initializing Alibaba AI Core ---")
    try:
        alibaba_rag_service.initialize()
    except Exception as e:
        print(f"Alibaba AI Init failed: {e}")
    
    print("\n[STARTUP] SUCCESS: All AI Cores are ready!")
    print("="*50 + "\n")
    yield
    print("[SHUTDOWN] Cleaning up memory...")

app = FastAPI(title="PA Chatbot Enterprise API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

@app.get("/")
async def root():
    return {"status": "ok", "service": "PA Chatbot API"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

