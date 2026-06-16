import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 랭체인 대신 구글 공식 생성형 AI SDK 사용
import google.generativeai as genai

# 1. API 키 설정 및 원본 SDK 초기화
#API_KEY = "AIzaSyC7_UMRV51iHGrAFtDGTAyPraZTBXrEfb0"
#os.environ["GOOGLE_API_KEY"] = API_KEY
#genai.configure(api_key=API_KEY)

import os
import google.generativeai as genai

# 환경 변수에서 API 키를 불러옵니다.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# 2. FastAPI 앱 초기화
app = FastAPI(title="Gemini AI Agent Server")

# 3. CORS 허용 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================================
# 💡 [핵심 수정] 4. 내 API 키로 사용 가능한 모델 자동 검색 및 설정
# =================================================================
available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
print(f"✅ 사용 가능한 모델 목록: {available_models}") # Render 로그에서 확인 가능

# 사용 가능한 모델 중 가장 빠르고 최신인 'flash' 모델을 자동으로 찾아냅니다.
target_model_name = available_models[0] # 만약을 대비한 기본값
for m_name in available_models:
    if "flash" in m_name:
        target_model_name = m_name

# 구글 API 규칙에 맞춰 'models/' 글자가 있으면 깔끔하게 떼어냅니다.
MODEL_NAME = target_model_name.replace("models/", "")
print(f"🚀 최종 연결된 모델: {MODEL_NAME}")

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction="당신은 구글 클라우드와 파이썬 기술을 지원하는 친절하고 유능한 AI 전문가 에이전트입니다. 답변은 명확하고 단계별로 제공하세요."
)
# =================================================================
# 4. 가장 호환성이 높은 모델로 에이전트 선언
# 구형 API 대역(v1beta)에서도 100% 작동하는 명칭입니다.
#MODEL_NAME = "gemini-1.5-flash"
#model = genai.GenerativeModel(
#    model_name=MODEL_NAME,
#    system_instruction="당신은 구글 클라우드와 파이썬 기술을 지원하는 친절하고 유능한 AI 전문가 에이전트입니다. 답변은 명확하고 단계별로 제공하세요."
#)

# 5. 요청 데이터 구조 정의
class UserRequest(BaseModel):
    message: str

# 6. 기본 연결 테스트 엔드포인트
@app.get("/")
def read_root():
    return {"status": "healthy", "agent": "Gemini ready"}

# 7. AI 에이전트 엔드포인트
@app.post("/chat")
async def chat_endpoint(request: UserRequest):
    try:
        # 구글 원본 SDK 방식으로 제미나이 호출
        response = model.generate_content(request.message)

        # React 클라이언트가 수신할 수 있도록 'reply' 키에 담아 반환
        return {"reply": response.text}

    except Exception as e:
        print(f"❌ 백엔드 에러 발생 원인: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
     
