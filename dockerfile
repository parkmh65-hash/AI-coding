ROM python:3.12-slim

WORKDIR /multi/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn","multi.app:app","--host","0.0.0.0","--port","8080"]
