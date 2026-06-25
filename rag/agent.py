"""
agent.py
────────
세 가지 도구(시간 / 웹검색 / 유튜브) + RAG 검색을 함께 처리하는 에이전트.

실행 흐름:
  1. RAG retriever 로 관련 문서 검색
  2. LLM + 도구 바인딩으로 답변 생성 (tool_call 사이클 최대 5회)
  3. 최종 텍스트 답변 + 도구 사용 로그 반환
"""

import os
from datetime import datetime
from typing import List

import pytz
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from youtube_search import YoutubeSearch
from langchain_community.document_loaders import YoutubeLoader

import retriever as rag   # RAG 모듈

# ── 모델 초기화 ───────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini")

# ── 도구 정의 ─────────────────────────────────────────────────
@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수."""
    try:
        tz  = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f"{timezone} ({location}) 현재시각 {now}"
        print(result)
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"


@tool
def get_web_search(query: str, search_period: str) -> str:
    """
    웹 검색을 수행하는 함수.

    Args:
        query (str): 검색어
        search_period (str): 검색 기간 ("w"=1주, "m"=1달, "y"=1년)

    Returns:
        str: 검색 결과
    """
    print("-------- WEB SEARCH --------")
    print(query, search_period)
    wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time=search_period)
    search  = DuckDuckGoSearchResults(api_wrapper=wrapper, results_separator=";\n")
    return search.invoke(query)


@tool
def get_youtube_search(query: str) -> List:
    """
    유튜브 검색 후 영상 내용(자막)을 반환하는 함수.

    Args:
        query (str): 검색어

    Returns:
        List: 자막이 포함된 영상 목록
    """
    print("-------- YOUTUBE SEARCH --------")
    print(query)
    videos = YoutubeSearch(query, max_results=5).to_dict()
    # 1시간 미만 영상만 처리 (duration 문자열 길이 5 이하 = "MM:SS")
    videos = [v for v in videos if len(v["duration"]) <= 5]
    for video in videos:
        video_url = "http://youtube.com" + video["url_suffix"]
        loader = YoutubeLoader.from_youtube_url(video_url, language=["ko", "en"])
        video["video_url"] = video_url
        video["content"]   = loader.load()
    return videos


# ── 도구 등록 ─────────────────────────────────────────────────
tools     = [get_current_time, get_web_search, get_youtube_search]
tool_dict = {t.name: t for t in tools}

llm_with_tools = llm.bind_tools(tools)


# ── RAG + 에이전트 통합 프롬프트 ─────────────────────────────
SYSTEM_TEMPLATE = """너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다.

아래는 문서에서 검색한 관련 내용이다. 질문에 관련이 있으면 참고하여 답변하라:
{rag_context}
"""


# ── 핵심 실행 함수 ────────────────────────────────────────────
def run(query: str, history: list) -> dict:
    """
    GAS에서 전달받은 query + history를 처리하여
    최종 답변과 도구 로그를 반환합니다.

    Args:
        query   (str) : 사용자의 최신 질문
        history (list): [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        dict: {"answer": str, "tool_log": [...], "augmented_query": str}
    """
    # ── GAS 히스토리 → LangChain 메시지 변환 ─────────────────
    lc_history = []
    for msg in history:
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))

    # ── ① RAG: 질문 정제 → 문서 검색 ────────────────────────
    augmented_query = rag.query_augmentation_chain.invoke({
        "messages": lc_history,
        "query": query,
    })
    docs = rag.retriever.invoke(f"{query}\n{augmented_query}")

    # 검색된 문서를 텍스트로 합치기
    rag_context = "\n\n".join(
        [f"[문서 {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
    ) if docs else "관련 문서 없음"

    # ── ② 에이전트 메시지 구성 ───────────────────────────────
    system_msg = SystemMessage(SYSTEM_TEMPLATE.format(rag_context=rag_context))
    lc_messages = [system_msg] + lc_history + [HumanMessage(content=query)]

    # ── ③ 도구 사이클 실행 ───────────────────────────────────
    tool_log = []
    _run_loop(lc_messages, tool_log)

    # 마지막 AIMessage 텍스트가 최종 답변
    for msg in reversed(lc_messages):
        if isinstance(msg, AIMessage) and msg.content:
            return {
                "answer": msg.content,
                "tool_log": tool_log,
                "augmented_query": augmented_query,
            }

    return {"answer": "(응답 없음)", "tool_log": tool_log, "augmented_query": augmented_query}


def _run_loop(messages: list, tool_log: list, depth: int = 0) -> None:
    """tool_call 사이클을 재귀 처리 (최대 5회)."""
    if depth >= 5:
        return

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        return  # 도구 호출 없음 → 종료

    for tool_call in response.tool_calls:
        name          = tool_call["name"]
        selected_tool = tool_dict[name]
        tool_msg      = selected_tool.invoke(tool_call)

        try:
            result_for_log = tool_msg.content
        except Exception:
            result_for_log = str(tool_msg)

        tool_log.append({"tool": name, "result": result_for_log})
        messages.append(tool_msg)

    _run_loop(messages, tool_log, depth + 1)
