"""
main.py
─────────────────────────────
FastAPI 서버
"""

import os

from fastapi import (
    FastAPI,
    HTTPException,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from pydantic import BaseModel

from typing import (
    List,
    Literal,
    Any,
)

api_key = os.getenv(
    "OPENAI_API_KEY"
)

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY가 없습니다."
    )

os.environ["OPENAI_API_KEY"] = api_key

import agent


app = FastAPI(
    title="GPT Agent API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: Literal[
        "user",
        "assistant"
    ]
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


@app.get("/health")
async def health():
    return {
        "status": "ok"
    }


@app.get("/")
async def root():
    return {
        "message": "server running"
    }


@app.post(
    "/chat",
    response_model=ChatResponse
)
async def chat(
    request: ChatRequest
):

    try:

        history = [
            {
                "role": m.role,
                "content": m.content,
            }
            for m in request.messages
        ]

        result = agent.run(
            request.query,
            history,
        )

        return ChatResponse(
            answer=result["answer"],
            augmented_query=result.get(
                "augmented_query",
                ""
            ),
            tool_log=result.get(
                "tool_log",
                []
            ),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000,
        )
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
