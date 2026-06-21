import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# 💡 신형 라이브러리 임포트
from google import genai

from google.genai import types
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

# 사용자 정의 함수(tools) 불러오기
from gpt_functions import get_current_time, tools, get_yf_stock_info, get_yf_stock_history, get_yf_stock_recommendations

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Gemini AI Agent Server")

# GAS 등 외부 클라이언트에서 API를 호출할 수 있도록 CORS 설정 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=api_key)

# 클라이언트로부터 받을 데이터 모델 정의 (대화 기록 전체를 받음)
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]

#class UserRequest(BaseModel):
#    message: str

@app.get("/")
def read_root():
    return {"status": "healthy", "agent": "Gemini ready"}

@app.post("/chat")
async def chat_endpoint(request: UserRequest):
    try:
        # 1. 환경 변수에서 새 API 키를 불러와 클라이언트 초기화
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API 키가 환경 변수에 없습니다.")
            
        client = genai.Client(api_key=api_key)

        # 2. 신형 SDK 방식으로 모델 호출 및 시스템 프롬프트 설정
        response = client.models.generate_content(
            model='gemini-2.5-flash', # ✅ 이렇게 1.5로 변경해 주세요!
            contents=request.message,
            config=types.GenerateContentConfig(
                system_instruction="당신은 구글 클라우드와 파이썬 기술을 지원하는 친절하고 유능한 AI 전문가 에이전트입니다. 답변은 명확하고 단계별로 제공하세요.",
            )
        )
        return {"reply": response.text}

    except Exception as e:
        print(f"❌ 백엔드 에러 발생 원인: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stock")
async def chat_endpoint(request: ChatRequest):
    try:
        messages = request.messages
        
        # 1. 사용자의 전체 대화 기록을 바탕으로 OpenAI API 1차 호출
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        ai_message = response.choices[0].message
        
        # 2. AI가 도구(함수) 호출이 필요하다고 판단한 경우
        if ai_message.tool_calls:
            # 🚨 중요: 도구를 호출한 AI의 응답 자체도 대화 기록에 추가해야 에러가 나지 않음
            messages.append(ai_message.model_dump(exclude_unset=True))
            
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id
                arguments = json.loads(tool_call.function.arguments)
                
                # 함수 실행
                if tool_name == "get_current_time":  
                    func_result = get_current_time(timezone=arguments.get('timezone'))
                elif tool_name == "get_yf_stock_info":
                    func_result = get_yf_stock_info(ticker=arguments.get('ticker'))
                elif tool_name == "get_yf_stock_history":
                    func_result = get_yf_stock_history(
                        ticker=arguments.get('ticker'), 
                        period=arguments.get('period')
                    )
                elif tool_name == "get_yf_stock_recommendations":
                    func_result = get_yf_stock_recommendations(
                        ticker=arguments.get('ticker')
                    )
                else:
                    func_result = "지원하지 않는 함수입니다."

                # 3. 함수 실행 결과를 역할(role)="tool"로 지정하여 대화 기록에 추가
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": str(func_result),
                })
            
            # 4. 함수 결과를 바탕으로 최종 답변 생성을 위한 2차 OpenAI API 호출
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
            )
            ai_message = response.choices[0].message

        # 최종적으로 생성된 텍스트 응답만 클라이언트(React)로 반환
        return {"role": "assistant", "content": ai_message.content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
