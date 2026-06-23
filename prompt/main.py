import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types
from openai import OpenAI

import json
from dotenv import load_dotenv

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from collections import defaultdict

# 사용자 정의 함수(tools) 불러오기
from .gpt_functions import get_current_time, tools, get_yf_stock_info, get_yf_stock_history, get_yf_stock_recommendations
# from gpt_functions import ... 대신 아래와 같이 점(.)을 사용
#from .gpt_functions import get_current_time, tools, ... 

#import sys
#import os
# 루트 디렉토리를 경로에 추가하여 인식하게 함
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from gpt_functions import get_current_time, tools, ...

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기

# 🚨 Render 배포 시 환경 변수 누락을 방지하고 명확한 에러를 띄우기 위한 안전장치
if not api_key:
    raise ValueError("🚨 OPENAI_API_KEY가 설정되지 않았습니다! Render 대시보드의 'Environment' 탭에서 환경 변수를 반드시 등록해주세요.")

# 🚨 추가된 핵심 해결책: Render 대시보드에 키를 입력할 때 실수로 들어간 줄바꿈, 공백, 'pip' 글자를 코드가 알아서 완벽하게 청소합니다!
api_key = api_key.replace('\r', '').replace('\n', '').replace('pip', '').strip()


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


client = OpenAI(api_key=api_key)  # 오픈AI 클라이언트의 인스턴스 생성

# 작성해주신 Tool Chunk 조립 함수
def tool_list_to_tool_obj(tools_chunk_list):
    # 기본 값을 가진 딕셔너리 초기화
    tool_calls_dict = defaultdict(lambda: {"id": None, "function": {"arguments": "", "name": None}, "type": None})

    # 도구(함수) 호출을 반복하여 처리
    for tool_call in tools_chunk_list:
        # id가 None이 아닌 경우 설정
        if tool_call.id is not None:
            tool_calls_dict[tool_call.index]["id"] = tool_call.id

        # 함수 이름이 None이 아닌 경우 설정
        if tool_call.function.name is not None:
            tool_calls_dict[tool_call.index]["function"]["name"] = tool_call.function.name

        # 인수 추가 (chunk 단위로 들어오므로 문자열 연결)
        if tool_call.function.arguments is not None:
            tool_calls_dict[tool_call.index]["function"]["arguments"] += tool_call.function.arguments

        # 타입이 None이 아닌 경우 설정
        if tool_call.type is not None:
            tool_calls_dict[tool_call.index]["type"] = tool_call.type

    # 딕셔너리를 리스트로 변환
    tool_calls_list = list(tool_calls_dict.values())
    return {"tool_calls": tool_calls_list}  

# 스트리밍을 지원하는 OpenAI 호출 함수
def get_ai_response(messages, tools=None, stream=True):
    response = client.chat.completions.create(
        model="gpt-4o",  
        stream=stream, 
        messages=messages,  
        tools=tools,  
    )
    if stream: 
        for chunk in response:
            yield chunk  
    else:
        return response  

# 클라이언트로부터 받을 대화 모델 정의
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]

@app.post("/stock")
async def chat_endpoint(request: ChatRequest):
    try:
        messages = request.messages
        
        # 1. 사용자의 전체 대화 기록을 바탕으로 스트리밍 호출
        ai_response = get_ai_response(messages, tools=tools, stream=True)
        
        content = ''
        tool_calls_chunk = []   
        
        print("\n[AI 응답 스트리밍]: ", end="")
        for chunk in ai_response:
            if not chunk.choices: continue
            delta = chunk.choices[0].delta
            
            if delta.content: 
                print(delta.content, end="") 
                content += delta.content 
            
            if delta.tool_calls:
                tool_calls_chunk += delta.tool_calls 

        tool_calls = []
        if tool_calls_chunk:
            tool_obj = tool_list_to_tool_obj(tool_calls_chunk)
            tool_calls = tool_obj["tool_calls"]   

        # 2. 도구 호출이 판단된 경우
        if len(tool_calls) > 0: 
            print("\n[호출된 도구 목록]:", tool_calls)
            
            # 🚨 중요: 도구를 호출한 AI의 결정 자체를 대화 기록에 넣어야 오류가 나지 않음
            messages.append({
                "role": "assistant",
                "content": content if content else None,
                "tool_calls": tool_calls
            })

            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]  
                tool_call_id = tool_call["id"]         
                arguments = json.loads(tool_call["function"]["arguments"])     
                
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
                    func_result = "지원하지 않는 기능입니다."

                # 🚨 주의: 최근 OpenAI API 규칙에 맞춰 role을 "function"이 아닌 "tool"로 고정합니다.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": str(func_result),
                })

            messages.append({
                "role": "system", 
                "content": "이제 주어진 결과를 바탕으로 답변할 차례다."
            }) 
            
            # 3. 함수 결과를 바탕으로 최종 답변을 다시 스트리밍
            ai_response2 = get_ai_response(messages, tools=tools, stream=True) 
            content = ""
            print("\n[AI 최종 답변 스트리밍]: ", end="")
            for chunk in ai_response2:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end='')
                    content += delta.content

        print("\n====================")
        
        # 클라이언트(React)에는 하나로 합쳐진 최종 텍스트만 전달합니다.
        return {"role": "assistant", "content": content}
        
    except Exception as e:
        print("서버 에러:", e)
        raise HTTPException(status_code=500, detail=str(e))
