FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Đảm bảo app nằm trong python path
ENV PYTHONPATH=/app

# Chạy trực tiếp uvicorn từ command line để ổn định nhất
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
