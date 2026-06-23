import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage

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
