FROM python:3.12-slim

WORKDIR /app

COPY multi/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn multi.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
