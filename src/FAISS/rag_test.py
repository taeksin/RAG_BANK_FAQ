import os
import time
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# 전처리 이전
FAISS_FOLDER = "data/faiss_index"
PROMPT_TEXT = "src/FAISS/prompt.txt"
# PROMPT_TEXT = "src/FAISS/prompt_before.txt"

# 전처리 이후
# FAISS_FOLDER = "data/faiss_index_clean"
# PROMPT_TEXT = "src/FAISS/prompt.txt"

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# 임베딩 모델 생성
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# FAISS 저장소 경로 설정
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
faiss_path = os.path.join(base_dir, f"{FAISS_FOLDER}")

# 벡터 저장소 로드
vectorstore = FAISS.load_local(faiss_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 10, 'lambda_mult': 0.9})

# 프롬프트 파일 경로 설정
prompt_file_path = os.path.join(base_dir, f"{PROMPT_TEXT}")

# 텍스트 파일을 읽어와 프롬프트 텍스트로 저장
def load_prompt_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# 파일에서 프롬프트 텍스트 읽기
prompt_text = load_prompt_from_file(prompt_file_path)

# PromptTemplate 설정
prompt = PromptTemplate(
    input_variables=["question", "context"],  # 필요한 입력 변수 설정
    template=prompt_text  # 텍스트 파일에서 읽어온 프롬프트 텍스트
)

# LLM 모델 설정
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)

# RAG 체인 설정
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # 질의 및 컨텍스트 설정
    | prompt  # 읽어온 텍스트 파일을 템플릿으로 사용
    | llm  # OpenAI의 GPT 모델을 사용하여 응답 생성
    | StrOutputParser()  # 문자열로 출력
)

# 메인 루프
if __name__ == "__main__":
    while True:
        try:
            user_input = input("질문을 입력하세요 (종료하려면 'exit' 입력): ")
            if user_input.lower() == "exit":
                print("챗봇을 종료합니다.")
                break

            # RAG 응답 생성
            retrieved_documents = retriever.invoke(user_input)
            response = rag_chain.invoke(user_input)

            print(f"응답: {response}")
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
