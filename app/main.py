import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import controllers
from app.controllers import chat_controller

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Không làm gì nặng nề ở đây
    print("--- SERVER STARTING ---")
    yield
    # Shutdown
    print("--- SERVER SHUTTING DOWN ---")

app = FastAPI(
    title="PA Chatbot API",
    lifespan=lifespan
)

# CORS cực kỳ quan trọng cho demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thêm router
app.include_router(chat_controller.router)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "PA Chatbot Backend is running",
        "endpoints": ["/docs", "/api/v1/chat"]
    }
