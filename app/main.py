import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.endpoints import chat, procedures
from app.services.rag_service import rag_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("=" * 50)
    print("[STARTUP] INITIALIZING ALIBABA CLOUD RAG SYSTEM...")
    print("=" * 50)
    try:
        # Chỉ khởi tạo một core duy nhất dùng Alibaba Cloud
        rag_service.initialize()
        print("[STARTUP] SUCCESS: RAG System is ready for production!")
    except Exception as e:
        print(f"[STARTUP] ERROR during initialization: {e}")
    print("=" * 50)
    yield
    # Shutdown logic
    print("[SHUTDOWN] Cleaning up resources...")

app = FastAPI(
    title="PA Chatbot API (Alibaba Cloud)",
    description="Backend API for PA Chatbot using Alibaba DashScope and Qdrant Cloud",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(procedures.router, prefix="/api/v1", tags=["Procedures"])

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "PA Chatbot Backend",
        "engine": "Alibaba DashScope (Qwen-Max)",
        "database": "Qdrant Cloud"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Production không nên dùng reload=True, nhưng để demo có thể để tạm
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
