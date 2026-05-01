from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok"}

# Chỉ import router nếu không phải là môi trường build
if os.environ.get("RAILWAY_GIT_COMMIT_SHA"):
    try:
        from app.controllers import chat_controller
        app.include_router(chat_controller.router)
    except Exception as e:
        print(f"Deferred router load error: {e}")
