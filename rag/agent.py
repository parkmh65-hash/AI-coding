"""
agent.py
───────────────────────────────────────────────
RAG + Tool Agent 통합 버전 (Render 최적화)

변경사항
- Lazy Loading 적용
- import 시 Chroma 생성 안함
- import 시 PDF 로딩 안함
- import 시 Embedding 생성 안함
- 첫 질문 시 RAG 초기화
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List

import pytz

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from langchain_core.tools import tool

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.output_parsers import StrOutputParser

from langchain_chroma import Chroma

from langchain_community.document_loaders import (
    PyPDFLoader,
    YoutubeLoader,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_community.tools import (
    DuckDuckGoSearchResults,
)

from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
)

from youtube_search import YoutubeSearch


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini")

rag_llm = None
embedding = None

vectorstore = None
retriever = None

# ============================================================
# PATH
# ============================================================

persist_directory = os.getenv(
    "CHROMA_PERSIST_DIR",
    "/tmp/chroma_store"
)

BASE_DIR = Path(__file__).resolve().parent
data_directory = BASE_DIR / "data"


# ============================================================
# Lazy Loading
# ============================================================

def get_embedding():
    global embedding

    if embedding is None:
        print("Embedding 초기화")
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )

    return embedding


def get_rag_llm():
    global rag_llm

    if rag_llm is None:
        rag_llm = ChatOpenAI(model="gpt-4o")

    return rag_llm


# ============================================================
# VectorStore
# ============================================================

def build_vectorstore():

    print("새로운 Vector DB 생성")

    documents = []

    if not data_directory.exists():
        raise RuntimeError(
            f"{data_directory} 폴더가 존재하지 않습니다."
        )

    for pdf_file in data_directory.glob("*.pdf"):

        print(f"PDF 로딩: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))

        documents.extend(loader.load())

    if not documents:
        raise RuntimeError("PDF 문서가 없습니다.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    texts = splitter.split_documents(documents)

    print(f"총 청크 수: {len(texts)}")

    return Chroma.from_documents(
        documents=texts,
        embedding=get_embedding(),
        persist_directory=persist_directory,
    )


def chroma_exists():

    if not os.path.isdir(persist_directory):
        return False

    for root, _, files in os.walk(persist_directory):
        for file in files:
            if file.endswith(
                (
                    ".sqlite3",
                    ".parquet",
                    ".bin",
                )
            ):
                return True

    return False


def init_rag():

    global vectorstore
    global retriever

    if retriever is not None:
        return

    print("================================")
    print("RAG 초기화 시작")
    print("================================")

    emb = get_embedding()

    if chroma_exists():

        print("기존 Chroma DB 로드")

        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=emb,
        )

    else:

        print("새 Chroma DB 생성")

        vectorstore = build_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    print("================================")
    print("RAG 초기화 완료")
    print("================================")


# ============================================================
# Query Augmentation
# ============================================================

query_augmentation_prompt = (
    ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(
                variable_name="messages"
            ),
            (
                "system",
                """
질문의 의도를 파악하여
검색에 적합한 한 문장으로 변환하라.

{query}
"""
            ),
        ]
    )
)


def get_query_chain():

    return (
        query_augmentation_prompt
        | get_rag_llm()
        | StrOutputParser()
    )


# ============================================================
# TOOLS
# ============================================================


@tool
def get_current_time(
    timezone: str,
    location: str,
) -> str:
    """
    현재 시간을 반환하는 도구.

    Args:
        timezone:
            pytz 형식의 타임존
            예: Asia/Seoul

        location:
            지역명

    Returns:
        현재 시간 문자열
    """

    try:

        tz = pytz.timezone(timezone)

        now = datetime.now(tz).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        return (
            f"{location} 현재 시각: {now}"
        )

    except Exception:

        return "알 수 없는 타임존"



@tool
def get_web_search(
    query: str,
    search_period: str,
) -> str:
    """
    인터넷 검색을 수행하는 도구.

    Args:
        query:
            검색할 질문 또는 키워드

        search_period:
            검색 기간
            w = 최근 1주
            m = 최근 1개월
            y = 최근 1년

    Returns:
        웹 검색 결과
    """

    wrapper = DuckDuckGoSearchAPIWrapper(
        region="kr-kr",
        time=search_period,
    )


    search = DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        results_separator=";\n",
    )


    return search.invoke(query)



@tool
def get_youtube_search(
    query: str,
) -> List:
    """
    유튜브 영상을 검색하고 자막을 가져오는 도구.

    Args:
        query:
            검색할 영상 주제

    Returns:
        영상 제목, URL, 자막 정보 목록
    """

    videos = YoutubeSearch(
        query,
        max_results=5,
    ).to_dict()


    results = []


    for video in videos:

        try:

            video_url = (
                "https://youtube.com"
                + video["url_suffix"]
            )


            loader = YoutubeLoader.from_youtube_url(
                video_url,
                language=["ko", "en"]
            )


            docs = loader.load()


            results.append(
                {
                    "title":
                        video.get("title"),

                    "url":
                        video_url,

                    "content":
                        docs[0].page_content[:3000]
                        if docs else ""
                }
            )


        except Exception:

            pass


    return results


# ============================================================
# TOOL 등록
# ============================================================

tools = [
    get_current_time,
    get_web_search,
    get_youtube_search,
]

tool_dict = {
    t.name: t
    for t in tools
}

llm_with_tools = llm.bind_tools(
    tools
)

# ============================================================
# SYSTEM
# ============================================================

SYSTEM_TEMPLATE = """
너는 사용자를 돕는 AI Assistant이다.

RAG 검색 결과:

{rag_context}

관련 내용이 있으면 반드시 활용하라.
"""

# ============================================================
# RUN
# ============================================================

def run(query: str, history: list):

    init_rag()

    lc_history = []

    for msg in history:

        if msg["role"] == "user":

            lc_history.append(
                HumanMessage(
                    content=msg["content"]
                )
            )

        elif msg["role"] == "assistant":

            lc_history.append(
                AIMessage(
                    content=msg["content"]
                )
            )

    augmented_query = (
        get_query_chain().invoke(
            {
                "messages": lc_history,
                "query": query,
            }
        )
    )

    docs = retriever.invoke(
        f"{query}\n{augmented_query}"
    )

    rag_context = "\n\n".join(
        [
            f"[문서 {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ]
    )

    if not rag_context:
        rag_context = "관련 문서 없음"

    messages = [
        SystemMessage(
            content=SYSTEM_TEMPLATE.format(
                rag_context=rag_context
            )
        )
    ]

    messages.extend(lc_history)
    messages.append(
        HumanMessage(content=query)
    )

    tool_log = []

    _run_loop(
        messages,
        tool_log,
    )

    for msg in reversed(messages):

        if (
            isinstance(msg, AIMessage)
            and msg.content
        ):
            return {
                "answer": msg.content,
                "tool_log": tool_log,
                "augmented_query": augmented_query,
            }

    return {
        "answer": "(응답 없음)",
        "tool_log": tool_log,
        "augmented_query": augmented_query,
    }


def _run_loop(
    messages,
    tool_log,
    depth=0,
):

    if depth >= 5:
        return

    response = llm_with_tools.invoke(
        messages
    )

    messages.append(response)

    if not response.tool_calls:
        return

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]

        tool_result = tool_dict[
            tool_name
        ].invoke(tool_call)

        tool_log.append(
            {
                "tool": tool_name,
                "result": str(tool_result),
            }
        )

        messages.append(tool_result)

    _run_loop(
        messages,
        tool_log,
        depth + 1,
    )
