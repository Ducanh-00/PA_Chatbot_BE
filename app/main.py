import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager

# Import controllers
from app.controllers import chat_controller

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- SERVER STARTING ---")
    yield
    print("--- SERVER SHUTTING DOWN ---")

app = FastAPI(
    title="PA Chatbot API",
    lifespan=lifespan
)

# Chấp nhận mọi host từ Railway
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]
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
    return {
        "status": "online",
        "message": "PA Chatbot Backend is running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    # Fallback cho chạy local
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
