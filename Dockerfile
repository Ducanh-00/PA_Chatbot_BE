FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY common/ ./common/

# Copy pre-built data (qdrant_db + BM25 cache)
COPY qdrant_db/ ./qdrant_db/
COPY all_chunks.pkl ./all_chunks.pkl

# Expose port
EXPOSE 8000

# Start server (Railway overrides this via railway.toml startCommand)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
