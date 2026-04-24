import os
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Thêm đường dẫn gốc để import thư mục common
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from common.enums.prompts import SYSTEM_PROMPT_BASIC_RAG
from common.enums.messages import CLIMessages

# 1. Tải biến môi trường (API Key)
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found. Please add it to the .env file")
    exit(1)

# 2. Configure Qdrant (Memory)
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "pa_procedures"
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"

print("Connecting to Qdrant DB...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
client = QdrantClient(path=QDRANT_PATH)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# Extract retriever from Vector DB
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. Initialize Gemini LLM (Brain)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# 4. Write Prompt Template (Instructions for AI)
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_BASIC_RAG),
    ("human", "{question}")
])

# Format documents for prompt
def format_docs(docs):
    formatted_text = ""
    for i, doc in enumerate(docs):
        category = doc.metadata.get('category', 'Unknown')
        source = doc.metadata.get('source_file', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        formatted_text += f"\n[Document {i+1} | Source: {source} - Page {page} | Category: {category}]\n{doc.page_content}\n"
    return formatted_text

# 5. Create RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Chat Loop
print(CLIMessages.BASIC_RAG_WELCOME)

while True:
    query = input(CLIMessages.PROMPT_USER)
    if query.lower() in ['exit', 'q', 'quit']:
        break
        
    print(CLIMessages.AI_THINKING)
    
    # Run AI
    response = rag_chain.invoke(query)
    
    print(f"AI: {response}")
    
    # Print citations for user verification
    print("\n" + "-"*40)
    print(CLIMessages.CITATIONS_HEADER)
    retrieved_docs = retriever.invoke(query)
    for doc in retrieved_docs:
        source = doc.metadata.get('source_file', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        print(f"  - {source} (Page {page})")
    print("-" * 40 + "\n")
