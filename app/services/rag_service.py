import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from flashrank import Ranker, RerankRequest

from app.core.config import settings

class RAGService:
    def __init__(self):
        self.rag_chain = None
        self.rewrite_chain = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.ranker = None
        
    def initialize(self):
        """Được gọi lúc startup server để load toàn bộ Model vào RAM."""
        print("1. Initialize Embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
        
        print("2. Initialize Vector DB...")
        client = QdrantClient(path=settings.QDRANT_PATH)
        vector_store = QdrantVectorStore(client=client, collection_name=settings.COLLECTION_NAME, embedding=embeddings)
        self.vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        
        print("3. Initialize BM25 Cache...")
        with open(settings.BM25_CACHE_PATH, "rb") as f:
            all_chunks = pickle.load(f)
        self.bm25_retriever = BM25Retriever.from_documents(all_chunks)
        self.bm25_retriever.k = 10
        
        print("4. Initialize Reranker...")
        self.ranker = Ranker(model_name=settings.RERANKER_MODEL_NAME)
        
        print("5. Initialize Gemini LLM & Chains...")
        llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL, temperature=0.2)
        
        from common.enums.prompts import REWRITE_QUESTION_PROMPT, SYSTEM_PROMPT_ADVANCED_RAG
        
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", REWRITE_QUESTION_PROMPT),
            ("human", "Question: {question}")
        ])
        self.rewrite_chain = rewrite_prompt | llm | StrOutputParser()
        
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT_ADVANCED_RAG), ("human", "Question: {question}")])
        self.rag_chain = prompt | llm | StrOutputParser()
        print("All AI logic loaded successfully!")

    def rewrite_question(self, question: str) -> str:
        return self.rewrite_chain.invoke({"question": question})

    def hybrid_search_and_rerank(self, query: str):
        # 1. Retrieve Dense + Sparse
        docs_dense = self.vector_retriever.invoke(query)
        docs_sparse = self.bm25_retriever.invoke(query)
        
        # 2. RRF Merging
        docs_map = {}
        for rank, doc in enumerate(docs_dense):
            docs_map[doc.page_content] = {"doc": doc, "score": 1 / (60 + rank)}
        for rank, doc in enumerate(docs_sparse):
            if doc.page_content in docs_map:
                docs_map[doc.page_content]["score"] += 1 / (60 + rank)
            else:
                docs_map[doc.page_content] = {"doc": doc, "score": 1 / (60 + rank)}
                
        combined_docs = sorted(docs_map.values(), key=lambda x: x["score"], reverse=True)[:10]
        
        # 3. Rerank
        passages = []
        for idx, item in enumerate(combined_docs):
            doc = item["doc"]
            passages.append({"id": str(idx), "text": doc.page_content, "meta": doc.metadata})
            
        rerankreq = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rerankreq)
        
        # 4. Format outputs
        final_docs = []
        for result in reranked_results[:3]:
            class Doc: pass
            d = Doc()
            d.page_content = result["text"]
            d.metadata = result["meta"]
            d.metadata["relevance_score"] = result["score"]
            final_docs.append(d)
        return final_docs

    def generate_answer(self, query: str, docs: list) -> str:
        context_text = ""
        for i, doc in enumerate(docs):
            context_text += f"\n[Document {i+1} | Source: {doc.metadata.get('source_file')} (Page {doc.metadata.get('page')})]\n{doc.page_content}\n"
        return self.rag_chain.invoke({"context": context_text, "question": query})

# Singleton instance
rag_service = RAGService()
