import os
import json
import streamlit as st
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# 페이지 설정
st.set_page_config(page_title="RAG 기반 챗봇", layout="wide")

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# Streamlit 기본 설정
st.title("RAG 기반 FAQ 챗봇 🤖")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "current_response" not in st.session_state:
    st.session_state.current_response = None
if "loading" not in st.session_state:
    st.session_state.loading = False

# JSON 파일 경로 설정
history_file_path = "chat_history.json"

# 히스토리 파일을 로드하는 함수
def load_chat_history():
    if os.path.exists(history_file_path):
        with open(history_file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return []

# 히스토리 저장 함수
def save_chat_history():
    try:
        with open(history_file_path, "w", encoding="utf-8") as file:
            json.dump(st.session_state.chat_history, file, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"히스토리 저장 중 오류가 발생했습니다: {e}")

# # 히스토리 디버깅: JSON 파일에서 로드된 데이터 확인
# st.write("디버깅: 로드된 히스토리:", st.session_state.chat_history)

# 임베딩 모델 생성
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# FAISS 저장소 경로 설정
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
faiss_path = os.path.join(base_dir, "data/faiss_index")

# 벡터 저장소 로드
vectorstore = FAISS.load_local(faiss_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 10, 'lambda_mult': 0.9})

# RAG 구성 요소 설정
prompt = hub.pull("fas_rag_platformdata")
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 히스토리 표시 함수
def display_history(container):
    with container:
        if st.session_state.chat_history:
            st.subheader("히스토리:")
            for i, entry in enumerate(st.session_state.chat_history):
                with st.expander(f"질문 {i+1}: {entry['질문']}"):
                    st.write(f"**질문:** {entry['질문']}")
                    st.write(f"**답변:** {entry['응답']}")

# UI 컨테이너 생성
history_container = st.container()

# 초기 히스토리 표시
display_history(history_container)

# 사용자 입력 처리
user_input = st.chat_input("질문을 입력하세요...")
if user_input:
    # 입력값 유효성 검사
    if not st.session_state.loading:
        st.session_state.current_question = user_input
        st.session_state.loading = True  # 로딩 상태 시작
        st.session_state.current_response = None

        # 현재 질문을 히스토리에 즉시 추가 (응답은 나중에 업데이트)
        st.session_state.chat_history.append({"질문": user_input, "응답": "응답 생성 중..."})
    else:
        st.warning("현재 질문에 대한 응답이 처리 중입니다. 잠시만 기다려주세요.")

# 기존 코드에서 응답 처리 부분
if st.session_state.loading and st.session_state.current_question:
    try:
        with st.spinner("답변을 생성 중입니다..."):
            # RAG 응답 생성
            retrieved_documents = retriever.invoke(st.session_state.current_question)
            response = rag_chain.invoke(st.session_state.current_question)

            # 응답 저장
            st.session_state.current_response = {
                "질문": st.session_state.current_question,
                "응답": response,
            }
            st.session_state.chat_history[-1]["응답"] = response

            # 히스토리 저장
            save_chat_history()

    except Exception as e:
        st.error(f"응답 생성 중 오류가 발생했습니다: {e}")
        st.session_state.chat_history[-1]["응답"] = "응답 생성에 실패했습니다. 다시 시도해주세요."
    finally:
        # 상태 갱신
        st.session_state.loading = False
        st.session_state.current_question = None

        # 응답 생성 후 히스토리 컨테이너를 새로 렌더링
        history_container.empty()  # 기존 내용을 지움
        display_history(history_container)  # 새로 렌더링

# 현재 응답 출력
if st.session_state.loading:
    st.subheader("챗봇 응답:")
    st.write("응답을 기다리는 중입니다...")

elif st.session_state.current_response:
    st.subheader("챗봇 응답:")
    st.write(f"**질문:** {st.session_state.current_response['질문']}")
    st.write(f"**응답:** {st.session_state.current_response['응답']}")
