from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
from langchain_openai import ChatOpenAI


# retriever 모듈은 별도 파일로 구성하여 import 하세요 [cite: 30]
# 현재 파일이 있는 디렉토리를 파이썬 경로에 추가
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import retriever 

app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    messages: List[dict] # 대화 기록을 클라이언트에서 받음

@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. 쿼리 증강 (Query Augmentation)
    augmented_query = retriever.query_augmentation_chain.invoke({
        "messages": request.messages,
        "query": request.query,
    })

    # 2. 문서 검색
    docs = retriever.retriever.invoke(f"{request.query}\n{augmented_query}")
    
    # 3. 답변 생성
    # stream 대신 전체 결과를 한 번에 받아 반환
    response = retriever.document_chain.invoke({
        "messages": request.messages,
        "context": docs
    })
    
    return {"answer": response, "augmented_query": augmented_query}
