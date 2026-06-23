from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pytz
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("에러: OPENAI_API_KEY가 설정되지 않았습니다.")
    
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
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"

# 2. 모델 및 도구 바인딩
llm = ChatOpenAI(
    model="gpt-4o-mini", # 밑줄(_)을 하이픈(-)으로 수정
    openai_api_key=api_key,
    request_timeout=60
)

tools = [get_current_time]
tool_dict = {"get_current_time": get_current_time}
llm_with_tools = llm.bind_tools(tools)

# 세션별 대화 기록 저장소
store = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "message": "API 서버가 실행 중입니다. /docs 에서 문서를 확인하세요."
    }
            
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    
    if session_id not in store:
        store[session_id] = [
            SystemMessage(content="너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다.")
        ]
    
    store[session_id].append(HumanMessage(content=request.message))
    
    while True:
        response = llm_with_tools.invoke(store[session_id])
        store[session_id].append(response)
        
        if not response.tool_calls:
            break
            
        for tool_call in response.tool_calls:
            selected_tool = tool_dict[tool_call['name']]
            tool_msg = selected_tool.invoke(tool_call)
            store[session_id].append(tool_msg)
            
    return {"role": "assistant", "content": response.content}
