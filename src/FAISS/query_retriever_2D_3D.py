import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# 시각화 차원 설정 (2D 또는 3D)
VISUALIZATION_DIM = 3  # 2 또는 3으로 변경 가능

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

# 시각화 함수 정의
def visualize_embeddings(question_embedding, retrieved_embeddings, k=5):
    """
    임베딩 값을 시각화합니다.

    :param question_embedding: 질문의 임베딩 값 (핑크색)
    :param retrieved_embeddings: 리트리버에서 검색된 임베딩 값 (fetch_k 중 k개는 빨간색, 나머지는 초록색)
    :param k: 최종 선택된 임베딩 개수
    """
    # 임베딩 데이터를 배열로 변환
    embeddings = np.array(retrieved_embeddings)
    question_point = np.array(question_embedding)

    if VISUALIZATION_DIM == 2:
        plt.figure(figsize=(10, 8))
        # fetch_k 중 k개의 임베딩 (빨간색)
        plt.scatter(embeddings[:k, 0], embeddings[:k, 1], c='red', label='Top-k Retrieved')
        # fetch_k 중 나머지 임베딩 (초록색)
        plt.scatter(embeddings[k:, 0], embeddings[k:, 1], c='green', label='Other Retrieved')
        # 질문 임베딩 (핑크색)
        plt.scatter(question_point[0], question_point[1], c='pink', label='Question')
        plt.legend()
        plt.title("Embedding Visualization (2D)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid()
        plt.show()

    elif VISUALIZATION_DIM == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        # fetch_k 중 k개의 임베딩 (빨간색)
        ax.scatter(embeddings[:k, 0], embeddings[:k, 1], embeddings[:k, 2], c='red', label='Top-k Retrieved')
        # fetch_k 중 나머지 임베딩 (초록색)
        ax.scatter(embeddings[k:, 0], embeddings[k:, 1], embeddings[k:, 2], c='green', label='Other Retrieved')
        # 질문 임베딩 (핑크색)
        ax.scatter(question_point[0], question_point[1], question_point[2], c='pink', label='Question', s=100)
        ax.legend()
        ax.set_title("Embedding Visualization (3D)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        plt.show()

# 질문 반복 처리
while True:
    question = input("\n 질문을 입력하세요 (종료하려면 'c', 'C' 또는 'ㅊ' 입력): ")
    if question.lower() in ["c", "ㅊ"]:
        print("Q&A 루프를 종료합니다.")
        break

    # 질문 임베딩 생성
    question_embedding = embedding_model.embed_query(question)

    # 리트리버에서 문서 검색
    retrieved_documents = retriever.invoke(question)

    # 검색된 문서의 임베딩 가져오기
    retrieved_embeddings = []
    for doc in retrieved_documents:
        # 새로운 메타데이터 구조로부터 임베딩 값 가져오기
        content_embedding = embedding_model.embed_query(doc.page_content)
        retrieved_embeddings.append(content_embedding)

    # RAG를 사용하여 응답 생성
    response = rag_chain.invoke(question)
    print("\n응답:", response)

    # 시각화 호출
    if retrieved_embeddings:
        visualize_embeddings(question_embedding, retrieved_embeddings, k=5)
    else:
        print("No embeddings available for visualization.")
