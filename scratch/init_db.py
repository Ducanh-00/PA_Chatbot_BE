import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from app.core.database import engine, Base
from app.models.chat import ChatSession, ChatMessage

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created successfully!")

if __name__ == "__main__":
    asyncio.run(init_db())
