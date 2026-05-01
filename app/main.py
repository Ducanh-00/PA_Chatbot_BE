import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Đảm bảo thư mục app có trong path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- SERVER STARTING ---")
    yield
    print("--- SERVER SHUTTING DOWN ---")

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

# Import router bên trong để tránh lỗi vòng lặp import hoặc crash sớm
try:
    from app.controllers import chat_controller
    app.include_router(chat_controller.router)
except Exception as e:
    print(f"FAILED TO LOAD ROUTER: {e}")

@app.get("/")
async def root():
    return {"status": "online", "port": os.environ.get("PORT", "8000")}

# Không dùng uvicorn.run ở đây nữa để Railway chạy qua CMD Dockerfile cho chuẩn
