import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.services.rag_service import rag_service
from app.controllers.chat_controller import router as chat_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Starting AI Core system...")
    rag_service.initialize()
    print("[STARTUP] Server is ready!")
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

