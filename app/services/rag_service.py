from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings

class RAGService:
    def __init__(self):
        self.rag_chain = None
        self.rewrite_chain = None
        self.vector_retriever = None
        
    def initialize(self):
        """Được gọi lúc startup server để load toàn bộ Model vào RAM."""
        try:
            print("1. Initialize Embeddings (Ollama Local)...")
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            
            print("2. Initialize Vector DB...")
            client = QdrantClient(path=settings.QDRANT_PATH)
            vector_store = QdrantVectorStore(client=client, collection_name=settings.COLLECTION_NAME, embedding=embeddings)
            self.vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            
            print("3. Initialize Gemini LLM & Chains...")
            llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL, temperature=0.2)
            
            from common.enums.prompts import REWRITE_QUESTION_PROMPT, SYSTEM_PROMPT_ADVANCED_RAG
            
            rewrite_prompt = ChatPromptTemplate.from_messages([
                ("system", REWRITE_QUESTION_PROMPT),
                ("human", "Question: {question}")
            ])
            self.rewrite_chain = rewrite_prompt | llm | StrOutputParser()
            
            prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT_ADVANCED_RAG), ("human", "Question: {question}")])
            self.rag_chain = prompt | llm | StrOutputParser()
            print("✅ All AI logic loaded successfully!")
        except Exception as e:
            print(f"❌ AI INITIALIZATION ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    def rewrite_question(self, question: str) -> str:
        return self.rewrite_chain.invoke({"question": question})

    def search(self, query: str):
        """Tìm kiếm tài liệu liên quan bằng Dense Vector Search."""
        print(f"[RAG] Searching for: {query}")
        docs = self.vector_retriever.invoke(query)
        print(f"[RAG] Found {len(docs)} relevant documents.")
        return docs

    def generate_answer(self, query: str, docs: list) -> str:
        context_text = ""
        for i, doc in enumerate(docs):
            context_text += f"\n[Document {i+1} | Source: {doc.metadata.get('source_file')} (Page {doc.metadata.get('page')})]\n{doc.page_content}\n"
        return self.rag_chain.invoke({"context": context_text, "question": query})

# Singleton instance
rag_service = RAGService()
