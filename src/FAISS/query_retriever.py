import os
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# 임베딩 모델 생성
embedding_model = OpenAIEmbeddings()

# 벡터 저장소 로드 (allow_dangerous_deserialization 인자를 추가)
vectorstore = FAISS.load_local("data/faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 20})

# RAG 구성 요소 설정 (프로젝트 맞춤 프롬프트와 LLM 설정)
prompt = hub.pull("fas_rag_platformdata")  # 맞춤형 프롬프트
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)  # LLM 설정
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 질문 반복 처리
while True:
    question = input("\n 질문을 입력하세요 (종료하려면 'c', 'C' 또는 'ㅊ' 입력): ")
    if question.lower() in ["c", "ㅊ"]:
        print("Q&A 루프를 종료합니다.")
        break
    
    # 리트리버에서 문서 검색
    retrieved_documents = retriever.invoke(question)

    # RAG를 사용하여 응답 생성
    response = rag_chain.invoke(question)
    print("\n응답:", response)
