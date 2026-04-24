FROM python:3.12-slim

WORKDIR /app

# Cài đặt system dependencies và dọn dẹp ngay để giảm dung lượng
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước
COPY requirements.txt .

# Cài đặt dependencies với CPU-only và không lưu cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn
COPY app/ ./app/
COPY common/ ./common/

# Copy dữ liệu đã build sẵn
COPY qdrant_db/ ./qdrant_db/
COPY all_chunks.pkl ./all_chunks.pkl

# Expose port
EXPOSE 8000

# Khởi động app (sử dụng shell form để mở rộng biến $PORT)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
