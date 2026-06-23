import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool # 이 라인이 반드시 있어야 합니다.

app = FastAPI()

# CORS 설정: GAS 웹앱 주소에서 호출 가능하도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 키 자동 정제 (Render 환경변수 줄바꿈 방지)
api_key = os.getenv("OPENAI_API_KEY", "").replace("\n", "").replace("\r", "").strip()
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

# 세션 관리 (메모리상)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain = RunnableWithMessageHistory(model, get_session_history)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message")
    session_id = data.get("session_id", "default_user")
    
    # AI 답변 생성 (invoke 사용)
    response = chain.invoke(
        [HumanMessage(content=user_input)], 
        config={"configurable": {"session_id": session_id}}
    )
    
    return {"reply": response.content}

#두번째 챗봇
llm2 = ChatOpenAI(model="gpt-4o-mini")

@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f'{timezone} ({location}) 현재시각 {now}'
    except:
        return f"알 수 없는 타임존: {timezone}"

tools = [get_current_time]
llm_with_tools = llm2.bind_tools(tools)

@app.post("/chat1")
async def chat(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    
    # 메시지 객체 복원
    formatted_messages = []
    for m in messages:
        if m["type"] == "human": formatted_messages.append(HumanMessage(m["content"]))
        elif m["type"] == "ai": formatted_messages.append(AIMessage(m["content"]))
        elif m["type"] == "tool": formatted_messages.append(ToolMessage(content=m["content"], tool_call_id=m["tool_call_id"]))

    response = llm_with_tools.invoke(formatted_messages)
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_result = get_current_time.invoke(tool_call)
        # 도구 실행 후 최종 응답 생성
        final_msg = llm_with_tools.invoke(formatted_messages + [response, ToolMessage(content=tool_result, tool_call_id=tool_call['id'])])
        return {"content": final_msg.content}
    
    return {"content": response.content}
    
