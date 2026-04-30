import os
from typing import List, Dict, Any
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rank_bm25 import BM25Okapi
import dashscope
from app.core.config import settings
from app.core.database import qdrant_manager

class RAGService:
    def __init__(self):
        self.vector_store = None
        self.llm = None
        self.embeddings = None
        self.bm25 = None
        self.all_docs = []
        self.is_initialized = False

    def initialize(self):
        if self.is_initialized:
            return
            
        print("--- [LAZY INITIALIZATION START] ---")
        dashscope.api_key = settings.DASHSCOPE_API_KEY
        dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

        print("1. Loading Alibaba Models...")
        self.embeddings = DashScopeEmbeddings(
            model=settings.ALIBABA_EMBEDDING_MODEL,
            dashscope_api_key=settings.DASHSCOPE_API_KEY
        )
        self.llm = ChatTongyi(
            model=settings.ALIBABA_CHAT_MODEL,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
            temperature=0.1
        )

        print("2. Connecting to Qdrant Cloud...")
        client = qdrant_manager.get_client()
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.ALIBABA_COLLECTION_NAME,
            embedding=self.embeddings,
        )

        print("3. Building BM25 Index for Hybrid Search...")
        try:
            # Fetch docs from Qdrant for BM25
            results = client.scroll(
                collection_name=settings.ALIBABA_COLLECTION_NAME,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            points = results[0]
            self.all_docs = [p.payload.get("page_content", "") for p in points if p.payload]
            
            if self.all_docs:
                tokenized_corpus = [doc.split() for doc in self.all_docs]
                self.bm25 = BM25Okapi(tokenized_corpus)
                print(f"   BM25 ready with {len(self.all_docs)} docs.")
            else:
                print("   Warning: No docs found for BM25.")
        except Exception as e:
            print(f"   Error building BM25: {e}")

        self.is_initialized = True
        print("--- [SUCCESS] RAG SYSTEM READY ---")

    def query(self, question: str) -> str:
        # Tự động khởi tạo nếu chưa xong
        if not self.is_initialized:
            self.initialize()

        # 1. Vector Search
        vector_results = self.vector_store.similarity_search(question, k=3)
        
        # 2. BM25 Search (Hybrid)
        bm25_context = ""
        if self.bm25:
            tokenized_query = question.split()
            bm25_hits = self.bm25.get_top_n(tokenized_query, self.all_docs, n=2)
            bm25_context = "\n".join(bm25_hits)

        # Combine Context
        context = "\n".join([doc.page_content for doc in vector_results])
        if bm25_context:
            context += "\n" + bm25_context

        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant for PA procedures. 
        Use the following context to answer the question. 
        If you don't know the answer, say you don't know.
        
        Context: {context}
        Question: {question}
        
        Answer (In Vietnamese):""")

        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(question)

rag_service = RAGService()
