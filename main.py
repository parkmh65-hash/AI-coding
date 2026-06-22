from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI()

# CORS 설정 (GAS 및 다양한 환경에서의 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션별 대화 기록을 저장할 전역 딕셔너리
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        # 세션이 처음 생성될 때 시스템 메시지 추가
        store[session_id].add_message(SystemMessage(content="너는 사용자의 질문에 친절히 답하는 AI챗봇이다."))
    return store[session_id]

# 모델 및 History 체인 설정 (API 키는 Render 환경변수에 등록되어 있다고 가정)
llm = ChatOpenAI(model="gpt-4o-mini")
with_message_history = RunnableWithMessageHistory(llm, get_session_history)

# 프론트엔드에서 받을 데이터 구조
class ChatRequest(BaseModel):
    session_id: str
    message: str

# 🚪 첫 번째 문: /chat 엔드포인트
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"session_id": request.session_id}}
    
    response = with_message_history.invoke(
        [HumanMessage(content=request.message)], 
        config=config
    )
    
    return {"role": "assistant", "content": response.content}

# 🚪 두 번째 문: /chat-1 엔드포인트 추가 (에러 해결의 핵심!)
@app.post("/chat-1")
def chat_endpoint_1(request: ChatRequest):
    # 로직은 위와 완벽하게 동일하지만, 주소가 다르므로 404 에러가 나지 않습니다.
    # 세션 ID가 다르기 때문에 대화 기록은 자동으로 완벽하게 분리됩니다.
    config = {"configurable": {"session_id": request.session_id}}
    
    response = with_message_history.invoke(
        [HumanMessage(content=request.message)], 
        config=config
    )
    
    return {"role": "assistant", "content": response.content}
