import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Read content from PDF file properly
loader = PyPDFLoader("./data/QĐ.KT.01 bản đi in.pdf")
docs = loader.load()

# 2. Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", ""],
    length_function=len,
)

# 3. Split the loaded documents
chunks = text_splitter.split_documents(docs)

# 4. Save chunks to a file in data_chunked
output_dir = "data_chunked"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "QD_KT_01_chunks.json")

chunks_data = [
    {
        "id": i,
        "page_content": chunk.page_content,
        "metadata": chunk.metadata
    }
    for i, chunk in enumerate(chunks)
]

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(chunks_data, f, ensure_ascii=False, indent=4)

print(f"Successfully saved {len(chunks)} chunks to {output_file}")
