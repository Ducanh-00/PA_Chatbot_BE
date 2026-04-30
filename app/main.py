import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Thêm dự án vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.controllers import chat_controller

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Không khởi tạo AI ở đây để server start nhanh
    print("=" * 50)
    print("[STARTUP] PA Chatbot API is online (Lazy Mode)")
    print("=" * 50)
    yield
    print("[SHUTDOWN] cleaning up...")

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

app.include_router(chat_controller.router)

@app.get("/")
async def root():
    return {"status": "online", "message": "Server is ready. AI will initialize on first request."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
