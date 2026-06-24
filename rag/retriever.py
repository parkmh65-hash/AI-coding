import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# FastAPI 앱 초기화 (Render에서 웹 서비스로 구동하기 위함)
app = FastAPI(title="LangChain RAG API on Render")

# 1. 임베딩 및 언어 모델 초기화
# (OPENAI_API_KEY는 Render의 Environment Variables 설정에서 추가해야 합니다)
embedding = OpenAIEmbeddings(model='text-embedding-3-large')
llm = ChatOpenAI(model="gpt-4o")

# 2. ChromaDB 로드 (Render 호환 경로 설정)
# C드라이브 절대경로 대신, 프로젝트 루트 기준의 상대 경로를 사용합니다.
# GitHub에 코드를 올릴 때 'chroma_store' 폴더도 같이 업로드되어야 작동합니다.
persist_directory = os.getenv('CHROMA_PERSIST_DIR', './chroma_store')

print(f"Loading existing Chroma store from: {persist_directory}")
vectorstore = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embedding
)

# Retriever 생성
retriever = vectorstore.as_retriever(k=3)

# 3. 체인 (Chain) 구성
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 질문에 대해 아래 context에 기반하여 답변하라.:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
document_chain = create_stuff_documents_chain(llm, question_answering_prompt) | StrOutputParser()

query_augmentation_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"), # 기존 대화 내용
        (
            "system",
            "기존의 대화 내용을 활용하여 사용자의 아래 질문의 의도를 파악하여 명료한 한 문장의 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라. :\n\n{query}",
        ),
    ]
)
query_augmentation_chain = query_augmentation_prompt | llm | StrOutputParser()

# 4. API 엔드포인트 정의
class ChatRequest(BaseModel):
    messages: list # 이전 대화 내용 리스트
    query: str     # 사용자의 현재 질문

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    클라이언트로부터 대화 기록과 질문을 받아 RAG 파이프라인을 거쳐 답변을 반환합니다.
    """
    # 1. 질문 의도 파악 및 변환
    augmented_query = query_augmentation_chain.invoke({
        "messages": request.messages,
        "query": request.query
    })
    
    # 2. 문서 검색
    # 최신 LangChain 버전에서는 get_relevant_documents 대신 invoke 사용을 권장합니다.
    docs = retriever.invoke(augmented_query)
    
    # 3. 답변 생성
    # 변환된 질문을 메시지 기록의 마지막에 추가하여 답변을 생성합니다.
    messages_with_query = request.messages + [("user", augmented_query)]
    
    answer = document_chain.invoke({
        "context": docs,
        "messages": messages_with_query
    })
    
    return {
        "augmented_query": augmented_query, 
        "answer": answer
    }

# 앱이 정상적으로 켜졌는지 확인하는 헬스체크용 엔드포인트
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Render RAG Server is running!"}
