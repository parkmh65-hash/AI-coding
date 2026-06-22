from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from datetime import datetime
import pytz

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

app = FastAPI()

# CORS 설정 (다양한 환경에서의 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 도구 함수 정의
@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f'{timezone} ({location}) 현재시각 {now}'
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"

# 모델 및 도구 초기화
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_current_time]
tool_dict = {"get_current_time": get_current_time}
llm_with_tools = llm.bind_tools(tools)

# 데이터 모델 정의
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    # 1. 프론트엔드 메시지를 LangChain 규격으로 변환
    lc_messages = [SystemMessage(content="너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다.")]
    
    for msg in request.messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=msg.content))

    # 2. 모델 호출 및 도구 실행 루프
    while True:
        response = llm_with_tools.invoke(lc_messages)
        lc_messages.append(response)

        # 도구 호출이 없으면 루프 종료
        if not response.tool_calls:
            break

        # 도구 실행 및 결과 추가
        for tool_call in response.tool_calls:
            selected_tool = tool_dict[tool_call['name']]
            tool_msg = selected_tool.invoke(tool_call)
            lc_messages.append(tool_msg)

    # 3. 최종 응답 반환
    return {"role": "assistant", "content": response.content}
