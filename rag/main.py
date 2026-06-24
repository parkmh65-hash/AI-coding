from fastapi import FastAPI, Request
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import os
from fastapi.middleware.cors import CORSMiddleware
import retriever  # 기존 RAG 로직 모듈

app = FastAPI()

# CORS 설정 (GAS에서 접근 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 설정
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
persist_directory = './chroma_store' # Render 배포 시 경로 주의

embedding = OpenAIEmbeddings(model='text-embedding-3-large')
llm = ChatOpenAI(model="gpt-4o")

# RAG 로직 (간소화)
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectorstore.as_retriever(k=3)

prompt = ChatPromptTemplate.from_messages([
    ("system", "아래 문맥을 참고하여 답변하라:\n\n{context}"),
    ("human", "{input}")
])

chain = create_stuff_documents_chain(llm, prompt)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_query = data.get("query")
    
    # 문서 검색
    docs = retriever.invoke(user_query)
    # RAG 답변 생성
    response = chain.invoke({"context": docs, "input": user_query})
    
    return {"reply": response}

async def chat_endpoint(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    user_query = messages[-1]["content"] # 마지막 메시지가 질문
    
    # 1. 질문 의도 파악 (Query Augmentation)
    augmented_query = retriever.query_augmentation_chain.invoke({
        "messages": messages,
        "query": user_query,
    })
    
    # 2. 관련 문서 검색
    docs = retriever.retriever.invoke(f"{user_query}\n{augmented_query}")
    
    # 3. 답변 생성
    response = retriever.document_chain.invoke({
        "messages": messages,
        "context": docs
    })
    
    return {"reply": response, "docs": [{"content": d.page_content, "meta": d.metadata} for d in docs]}





