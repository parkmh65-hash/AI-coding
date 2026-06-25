import sys
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# 1. 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. API 키 설정 (retriever import 전에 반드시 먼저 실행)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("환경 변수 OPENAI_API_KEY가 설정되지 않았습니다.")
os.environ["OPENAI_API_KEY"] = api_key.strip()

import retriever  # ✅ 키 설정 후 import

app = FastAPI()


class ChatRequest(BaseModel):
    query: str
    messages: List[dict]


@app.get("/health")
async def health():
    """Render 헬스체크용 엔드포인트"""
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # RAG 파이프라인
        augmented_query = retriever.query_augmentation_chain.invoke({
            "messages": request.messages,
            "query": request.query,
        })

        docs = retriever.retriever.invoke(f"{request.query}\n{augmented_query}")

        response = retriever.document_chain.invoke({
            "messages": request.messages,
            "context": docs,
        })

        return {"answer": response, "augmented_query": augmented_query}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 3. 서버 실행
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
