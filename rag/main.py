"""
main.py
───────
FastAPI 서버.
- OPENAI_API_KEY 설정을 import agent 보다 먼저 수행
- CORS 설정으로 GAS 웹앱 요청 허용
- /health  : Render 헬스체크 & 슬립 방지 핑
- /chat    : RAG + 에이전트 통합 응답
"""

import sys
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Any

# ── 1. 경로 + API 키 설정 (반드시 import agent 전에!) ─────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("환경 변수 OPENAI_API_KEY가 설정되지 않았습니다.")
os.environ["OPENAI_API_KEY"] = api_key.strip()

import agent   # 키 설정 완료 후 import
import retriever
# ── 2. FastAPI 앱 ─────────────────────────────────────────────
app = FastAPI(title="GPT-4o Langchain Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 필요시 GAS 도메인으로 제한 가능
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── 3. 스키마 ─────────────────────────────────────────────────
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    query: str
    messages: List[Message] = []

class ToolLog(BaseModel):
    tool: str
    result: Any

class ChatResponse(BaseModel):
    answer: str
    augmented_query: str = ""
    tool_log: List[ToolLog] = []

# ── 4. 엔드포인트 ─────────────────────────────────────────────
@app.get("/health")
async def health():
    """Render 헬스체크 & 슬립 방지 핑용"""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Pydantic 모델 → dict 변환 후 agent.run() 에 전달
        history = [{"role": m.role, "content": m.content} for m in request.messages]
        result  = agent.run(request.query, history)

        return ChatResponse(
            answer          = result["answer"],
            augmented_query = result.get("augmented_query", ""),
            tool_log        = result.get("tool_log", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 5. 로컬 실행 ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Render가 제공하는 포트를 사용하되, 없으면 기본값으로 8000 사용
    port = int(os.environ.get("PORT", 8000))
    # host는 반드시 "0.0.0.0"으로 설정해야 외부 접속이 가능합니다.
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
