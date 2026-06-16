import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# 💡 신형 라이브러리 임포트
from google import genai
from google.genai import types

app = FastAPI(title="Gemini AI Agent Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "healthy", "agent": "Gemini ready"}

@app.post("/chat")
async def chat_endpoint(request: UserRequest):
    try:
        # 1. 환경 변수에서 새 API 키를 불러와 클라이언트 초기화
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API 키가 환경 변수에 없습니다.")
            
        client = genai.Client(api_key=api_key)

        # 2. 신형 SDK 방식으로 모델 호출 및 시스템 프롬프트 설정
        response = client.models.generate_content(
            model='gemini-2.5-flash', # 최신 모델 명칭
            contents=request.message,
            config=types.GenerateContentConfig(
                system_instruction="당신은 구글 클라우드와 파이썬 기술을 지원하는 친절하고 유능한 AI 전문가 에이전트입니다. 답변은 명확하고 단계별로 제공하세요.",
            )
        )
        return {"reply": response.text}

    except Exception as e:
        print(f"❌ 백엔드 에러 발생 원인: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
