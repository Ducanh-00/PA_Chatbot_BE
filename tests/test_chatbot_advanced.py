import os
import sys
import pickle
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from flashrank import Ranker, RerankRequest

# Thêm đường dẫn gốc để import thư mục common
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from common.enums.prompts import REWRITE_QUESTION_PROMPT, SYSTEM_PROMPT_ADVANCED_RAG
from common.enums.messages import CLIMessages

load_dotenv()

QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "pa_procedures"
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"
ALL_CHUNKS_PATH = os.path.join(BASE_DIR, "all_chunks.pkl")

print("1. Initialize Models...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

print("2. Initialize Hybrid Retriever (Qdrant Vector + BM25 Local)...")
client = QdrantClient(path=QDRANT_PATH)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

with open(ALL_CHUNKS_PATH, "rb") as f:
    all_chunks = pickle.load(f)
bm25_retriever = BM25Retriever.from_documents(all_chunks)
bm25_retriever.k = 10

print("3. Initialize Re-ranker (Flashrank)...")
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")  # Model nhẹ chạy offline

print("4. Initialize Gemini LLM...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- MODULE 1: REWRITE QUESTION ---
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITE_QUESTION_PROMPT),
        ("human", "Original question: {question}"),
    ]
)
rewrite_chain = rewrite_prompt | llm | StrOutputParser()


# --- MODULE 2 & 3: CUSTOM HYBRID SEARCH & RERANK ---
def hybrid_search_and_rerank(query: str):
    # Get results from 2 sources
    docs_dense = vector_retriever.invoke(query)
    docs_sparse = bm25_retriever.invoke(query)

    # Merge using RRF (Reciprocal Rank Fusion)
    docs_map = {}
    for rank, doc in enumerate(docs_dense):
        docs_map[doc.page_content] = {"doc": doc, "score": 1 / (60 + rank)}
    for rank, doc in enumerate(docs_sparse):
        if doc.page_content in docs_map:
            docs_map[doc.page_content]["score"] += 1 / (60 + rank)
        else:
            docs_map[doc.page_content] = {"doc": doc, "score": 1 / (60 + rank)}

    # Get top 10 after merge
    combined_docs = sorted(docs_map.values(), key=lambda x: x["score"], reverse=True)[
        :10
    ]

    # Prepare data for Flashrank
    passages = []
    for idx, item in enumerate(combined_docs):
        doc = item["doc"]
        passages.append(
            {"id": str(idx), "text": doc.page_content, "meta": doc.metadata}
        )

    rerankreq = RerankRequest(query=query, passages=passages)
    reranked_results = ranker.rerank(rerankreq)

    # Return top 3
    final_docs = []
    for result in reranked_results[:3]:
        # Recreate metadata and content
        doc_meta = result["meta"]
        doc_meta["relevance_score"] = result["score"]

        # Fake Document object
        class Doc:
            pass

        d = Doc()
        d.page_content = result["text"]
        d.metadata = doc_meta
        final_docs.append(d)

    return final_docs


# --- MODULE 4: GENERATION ---
prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT_ADVANCED_RAG), ("human", "Question: {question}")]
)


def format_docs(docs):
    formatted_text = ""
    for i, doc in enumerate(docs):
        category = doc.metadata.get("category", "Unknown")
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "N/A")
        score = doc.metadata.get("relevance_score", 0)
        formatted_text += f"\n[Document {i + 1} | Source: {source} (Page {page}) | Re-rank Score: {score}]\n{doc.page_content}\n"
    return formatted_text


rag_chain = prompt | llm | StrOutputParser()

print(CLIMessages.ADVANCED_RAG_WELCOME)

while True:
    original_query = input(CLIMessages.PROMPT_USER)
    if original_query.lower() in ["exit", "q", "quit"]:
        break

    print(CLIMessages.ANALYZING)
    rewritten_query = rewrite_chain.invoke({"question": original_query})
    print(CLIMessages.REWRITTEN_TO.format(rewritten_query))

    print(CLIMessages.SEARCHING_HYBRID)
    try:
        retrieved_docs = hybrid_search_and_rerank(rewritten_query)

        print(CLIMessages.AI_GENERATING)
        context_text = format_docs(retrieved_docs)
        response = rag_chain.invoke(
            {"context": context_text, "question": rewritten_query}
        )

        print(f"Chatbot: \n{response}")

        print(CLIMessages.SOURCES_HEADER)
        for doc in retrieved_docs:
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "N/A")
            score = doc.metadata.get("relevance_score", 0)
            print(f"  - {source} (Page {page}) | Confidence Score: {score:.4f}")

    except Exception as e:
        print(CLIMessages.ERROR.format(e))
