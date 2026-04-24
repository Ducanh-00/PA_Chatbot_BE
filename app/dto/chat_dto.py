from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class SourceInfo(BaseModel):
    file: str
    page: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
