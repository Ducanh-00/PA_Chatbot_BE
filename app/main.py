import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import từ đúng cấu trúc thực tế của bạn
from app.controllers import chat_controller
from app.services.rag_service import rag_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("[STARTUP] INITIALIZING ALIBABA CLOUD RAG SYSTEM...")
    print("=" * 50)
    try:
        rag_service.initialize()
        print("[STARTUP] SUCCESS: RAG System is ready!")
    except Exception as e:
        print(f"[STARTUP] ERROR during initialization: {e}")
    print("=" * 50)
    yield
    print("[SHUTDOWN] Cleaning up resources...")

app = FastAPI(
    title="PA Chatbot API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thêm router từ chat_controller
app.include_router(chat_controller.router)

@app.get("/")
async def root():
    return {
        "status": "online", 
        "service": "PA Chatbot Backend",
        "docs": "/docs"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
