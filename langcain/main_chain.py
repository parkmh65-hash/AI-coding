from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pytz
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 도구 함수 정의
@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f'{timezone} ({location}) 현재시각 {now}'
        print(f"🛠️ [도구 실행됨] {result}")
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"

# 2. 모델 및 도구 바인딩
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_current_time]
tool_dict = {"get_current_time": get_current_time}
llm_with_tools = llm.bind_tools(tools)

# 세션별 대화 기록 저장소
store = {}

# 프론트엔드 요청 데이터 모델
class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    
    # 세션이 처음이면 시스템 메시지 추가
    if session_id not in store:
        store[session_id] = [
            SystemMessage(content="너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다.")
        ]
    
    # 사용자 메시지 저장
    store[session_id].append(HumanMessage(content=request.message))
    
    # 3. AI 응답 및 도구 실행 루프 (Streamlit의 재귀함수를 while문으로 안전하게 변환)
    while True:
        # 모델 호출
        response = llm_with_tools.invoke(store[session_id])
        store[session_id].append(response)
        
        # 도구 호출(tool_calls)이 없으면 루프를 종료하고 최종 답변 반환
        if not response.tool_calls:
            break
            
        # 도구 호출이 있다면 도구를 실행하고 결과를 다시 대화 기록에 추가
        for tool_call in response.tool_calls:
            selected_tool = tool_dict[tool_call['name']]
            tool_msg = selected_tool.invoke(tool_call)
            store[session_id].append(tool_msg)
            
    # 최종적으로 완성된 텍스트 응답만 프론트엔드에 전달
    return {"role": "assistant", "content": response.content}
