from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from .graph import graph
from .utils import (
    load_state,
    save_state
)

from datetime import datetime


app = FastAPI(
    title="Book Writer Agent"
)

# 헬스 체크 엔드포인트 추가
@app.get("/healthz")
def health_check():
    return {"status": "ok"}

class ChatRequest(BaseModel):

    user_input:str
    thread_id:str = "default"



@app.get("/")
def home():

    return {
        "status":"book writer running"
    }




@app.post("/chat")
def chat(
    req:ChatRequest
):


    state = load_state(
        req.thread_id
    )


    if state is None:

        state = {

            "messages":[

                SystemMessage(
f"""
너희는 AI 책 집필팀이다.

사용자의 언어로 대화한다.

현재시간:
{datetime.now()}
"""
                )

            ]

        }



    state["messages"].append(
        HumanMessage(
            req.user_input
        )
    )


    # LangGraph 실행

    result = graph.invoke(
        state
    )


    save_state(
        req.thread_id,
        result
    )


    answer = result["messages"][-1].content


    return {

        "response":answer,

        "message_count":
        len(result["messages"])

    }
