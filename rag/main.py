import sys
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Any

# ── 1. API 키 설정 (retriever import 전에 먼저!) ───────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("환경 변수 OPENAI_API_KEY가 설정되지 않았습니다.")
os.environ["OPENAI_API_KEY"] = api_key.strip()

import agent   # ← retriever 대신 agent 모듈

# ── 2. FastAPI 앱 ─────────────────────────────────────────────
app = FastAPI(title="GPT-4o Langchain Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    tool_log: List[ToolLog] = []

# ── 4. 엔드포인트 ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = agent.run(request.query, request.messages)
        return ChatResponse(
            answer=result["answer"],
            tool_log=result.get("tool_log", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 5. 로컬 실행 ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
