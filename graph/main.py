import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# FastAPI 앱 초기화
app = FastAPI(title="LangGraph API for GAS")

# 헬스 체크 엔드포인트 추가
@app.get("/healthz")
def health_check():
    return {"status": "ok"}
    
# 환경 변수에서 OpenAI API 키 확인 (Render의 Environment Variables에 설정해야 함)
if "OPENAI_API_KEY" not in os.environ:
    print("경고: OPENAI_API_KEY가 환경 변수에 설정되지 않았습니다.")

# --- LangGraph 설정 (제공해주신 코드 적용) ---
model = ChatOpenAI(model="gpt-4o-mini")

class State(TypedDict):
    """상태 관리를 위한 딕셔너리 구조체"""
    messages: Annotated[list[str], add_messages]

graph_builder = StateGraph(State)

def generate(state: State):
    """상태를 기반으로 응답을 생성하는 노드"""
    return {"messages": [model.invoke(state["messages"])]}

graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "generate")
graph_builder.add_edge("generate", END)    

# 메모리(체크포인터)를 전역으로 하나만 생성하여 상태 유지
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# --- API 엔드포인트 설정 ---
class ChatRequest(BaseModel):
    user_input: str
    thread_id: str = "default_thread_id" # 스레드 ID를 통해 대화 문맥 유지

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    GAS에서 전달받은 메시지를 LangGraph로 처리하여 응답을 반환합니다.
    """
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # 그래프 실행
        events = graph.stream(
            {"messages": [HumanMessage(content=request.user_input)]}, 
            config,
            stream_mode="values"
        )
        
        final_state = None
        # 제너레이터에서 최종 상태를 가져옵니다
        for event in events:
            final_state = event
            
        if final_state and "messages" in final_state:
            # 최종 메시지 (AI의 응답) 추출
            ai_message = final_state["messages"][-1].content
            total_messages = len(final_state["messages"])
            
            return {
                "response": ai_message,
                "thread_id": request.thread_id,
                "message_count": total_messages
            }
        else:
            raise HTTPException(status_code=500, detail="응답 생성에 실패했습니다.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

@app.get("/")
def health_check():
    """Render 헬스 체크용 엔드포인트"""
    return {"status": "ok", "message": "LangGraph API가 정상적으로 실행 중입니다."}
