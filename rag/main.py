import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain_openai import ChatOpenAI

# 1. 경로 설정 (필수)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import retriever

app = FastAPI()

# 2. 키 정제 (보안 및 에러 방지)
# Render 환경 변수에 OPENAI_API_KEY가 설정되어 있어야 합니다.
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key.strip() 

class ChatRequest(BaseModel):
    query: str
    messages: List[dict]

@app.post("/chat")
async def chat(request: ChatRequest):
    # RAG 파이프라인 호출
    augmented_query = retriever.query_augmentation_chain.invoke({
        "messages": request.messages,
        "query": request.query,
    })

    docs = retriever.retriever.invoke(f"{request.query}\n{augmented_query}")
    
    response = retriever.document_chain.invoke({
        "messages": request.messages,
        "context": docs
    })
    
    return {"answer": response, "augmented_query": augmented_query}

# 3. 서버 실행 코드 추가
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
