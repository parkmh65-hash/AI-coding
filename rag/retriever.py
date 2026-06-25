"""
retriever.py
────────────
ChromaDB 벡터스토어 + RAG 체인 초기화 모듈.
- Render는 /tmp 만 쓰기 가능 → persist_directory 기본값을 /tmp/chroma_store 로 변경
- as_retriever(k=3) 문법 오류 수정 → search_kwargs={"k": 3}
- vectorstore가 None 일 때 retriever 생성 시도 방어 처리
"""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# ── 1. 설정 ───────────────────────────────────────────────────
# Render 무료 플랜은 /tmp 만 쓰기 가능
persist_directory = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma_store")
data_directory    = "./data"

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
llm       = ChatOpenAI(model="gpt-4o")


# ── 2. 벡터스토어 빌드 함수 ───────────────────────────────────
def build_vectorstore():
    print("데이터베이스를 새로 빌드합니다...")

    if not os.path.exists(data_directory):
        raise RuntimeError(
            f"'{data_directory}' 폴더가 없습니다. PDF 파일을 추가하세요."
        )

    documents = []
    for filename in os.listdir(data_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_directory, filename)
            print(f"  로딩: {filename}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())

    if not documents:
        raise RuntimeError(f"'{data_directory}' 폴더에 PDF 파일이 없습니다.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts    = splitter.split_documents(documents)
    print(f"  총 {len(texts)}개 청크 생성 완료")

    return Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=persist_directory,
    )


# ── 3. 벡터스토어 로드 or 빌드 ────────────────────────────────
def _chroma_exists() -> bool:
    """persist_directory 안에 실제 DB 파일이 있는지 확인."""
    if not os.path.isdir(persist_directory):
        return False
    for root, _, files in os.walk(persist_directory):
        for f in files:
            if f.endswith((".parquet", ".bin", ".sqlite3")):
                return True
    return False


if _chroma_exists():
    print("기존 데이터베이스를 불러옵니다.")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
    )
else:
    vectorstore = build_vectorstore()

# ✅ 수정: k=3 → search_kwargs={"k": 3}
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ── 4. RAG 체인 구성 ──────────────────────────────────────────
question_answering_prompt = ChatPromptTemplate.from_messages([
    ("system", "사용자의 질문에 대해 아래 context에 기반하여 답변하라.:\n\n{context}"),
    MessagesPlaceholder(variable_name="messages"),
])
document_chain = create_stuff_documents_chain(llm, question_answering_prompt) | StrOutputParser()

query_augmentation_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("system", "질문 의도를 파악하여 명료한 한 문장의 질문으로 변환하라.:\n\n{query}"),
])
query_augmentation_chain = query_augmentation_prompt | llm | StrOutputParser()
