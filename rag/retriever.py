import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# 1. 설정
persist_directory = os.getenv('CHROMA_PERSIST_DIR', './chroma_store')
data_directory = './data' 
embedding = OpenAIEmbeddings(model='text-embedding-3-large')
llm = ChatOpenAI(model="gpt-4o")

# 2. 자동 빌드 함수
def build_vectorstore():
    print("데이터베이스를 새로 빌드합니다...")
    if not os.path.exists(data_directory):
        print(f"경고: {data_directory} 폴더가 없습니다.")
        return None
        
    documents = []
    for filename in os.listdir(data_directory):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_directory, filename))
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    return Chroma.from_documents(
        documents=texts, 
        embedding=embedding, 
        persist_directory=persist_directory
    )

# 3. 로직 실행 (단 하나의 vectorstore를 정의)
if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
    vectorstore = build_vectorstore()
else:
    print("기존 데이터베이스를 불러옵니다.")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)

retriever = vectorstore.as_retriever(k=3)

# 4. 체인 구성 (기존 코드 유지)
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
