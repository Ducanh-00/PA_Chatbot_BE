import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Lazy init - chỉ tạo engine khi thực sự cần (không crash app lúc khởi động)
_engine = None
_AsyncSessionLocal = None

def _get_engine():
    global _engine, _AsyncSessionLocal
    if _engine is None:
        raw_db_url = os.environ.get("DATABASE_URL")
        if not raw_db_url:
            raise ValueError("DATABASE_URL chưa được cấu hình trong biến môi trường")
        DATABASE_URL = raw_db_url.strip()
        _engine = create_async_engine(DATABASE_URL, echo=False)
        _AsyncSessionLocal = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    return _engine, _AsyncSessionLocal

# Dùng trong các controller qua FastAPI Depends
async def get_db():
    _, SessionLocal = _get_engine()
    async with SessionLocal() as session:
        yield session

# Dùng trực tiếp trong chat_controller (không qua Depends)
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_db_session():
    _, SessionLocal = _get_engine()
    async with SessionLocal() as session:
        yield session

