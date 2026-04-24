FROM python:3.12-slim

WORKDIR /app

# Cài đặt dependencies cơ bản
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn
COPY app/ ./app/
COPY common/ ./common/

# Copy dữ liệu (Lưu ý: Bạn cần chạy ingestion lại để đồng bộ với Google Embeddings)
COPY qdrant_db/ ./qdrant_db/
COPY all_chunks.pkl ./all_chunks.pkl

EXPOSE 8080

# Chạy với PORT từ Railway
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
