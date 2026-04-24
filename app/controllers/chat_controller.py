from fastapi import APIRouter, HTTPException
from app.dto.chat_dto import ChatRequest, ChatResponse, SourceInfo
from app.services.rag_service import rag_service

router = APIRouter(prefix="/api/chat", tags=["Chat"])

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Bước 1: Viết lại câu hỏi
        rewritten_query = rag_service.rewrite_question(request.question)
        
        # Bước 2: Tìm kiếm & Rerank tài liệu
        retrieved_docs = rag_service.search(rewritten_query)
        
        # Bước 3: Tạo câu trả lời
        answer = rag_service.generate_answer(rewritten_query, retrieved_docs)
        
        # Bước 4: Trích xuất trích dẫn
        sources = []
        for doc in retrieved_docs:
            sources.append(SourceInfo(
                file=doc.metadata.get("source_file", "Unknown"),
                page=str(doc.metadata.get("page", "N/A")),
                score=round(doc.metadata.get("relevance_score", 0), 4)
            ))
            
        return ChatResponse(answer=answer, sources=sources)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
