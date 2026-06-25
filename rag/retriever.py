import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader # PDF 로더 예시
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 1. 환경 변수 및 설정
# 데이터베이스 폴더 경로
persist_directory = os.getenv('CHROMA_PERSIST_DIR', './chroma_store')

# 폴더가 존재하지 않으면 새로 생성하는 로직 추가
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
    print(f"Created new directory: {persist_directory}")

vectorstore = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embedding
)

data_directory = './data' # 원본 문서(PDF 등)를 넣어둘 폴더
embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# 2. 자동 빌드 함수
def build_vectorstore():
    print("데이터베이스를 새로 빌드합니다...")
    
    # data 폴더의 모든 PDF 파일 로드
    documents = []
    for filename in os.listdir(data_directory):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_directory, filename))
            documents.extend(loader.load())
    
    # 텍스트 분할 (청크 생성)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # 벡터 DB 생성 및 저장
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embedding, 
        persist_directory=persist_directory
    )
    print("데이터베이스 빌드 완료!")
    return vectorstore

# 3. 로직 실행
if not os.path.exists(persist_directory):
    # 폴더가 없으면 새로 빌드
    vectorstore = build_vectorstore()
else:
    # 이미 폴더가 있으면 기존 DB 로드
    print("기존 데이터베이스를 불러옵니다.")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)

retriever = vectorstore.as_retriever(k=3)


# 임베딩 모델 선언하기
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# 언어 모델 불러오기
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

# Load Chroma store
from langchain_chroma import Chroma
print("Loading existing Chroma store")
persist_directory = './rag/chroma_store'

vectorstore = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embedding
)

# Create retriever
retriever = vectorstore.as_retriever(k=3)

# Create document chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser # 문자열 출력 파서를 불러옵니다.

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

# query augmentation chain
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
