import os
import json
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
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

# 쿠키 관리 초기화
cookies = EncryptedCookieManager(prefix="faq_chatbot", password="secure-password")
if not cookies.ready():
    st.stop()

# 세션 상태 초기화 및 쿠키에서 데이터 로드
if "chat_history" not in st.session_state:
    # 쿠키에서 데이터 로드
    chat_history = cookies.get("chat_history")
    if chat_history:
        st.session_state.chat_history = json.loads(chat_history)
    else:
        st.session_state.chat_history = []  # 초기화 시 빈 리스트로 설정

if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "current_response" not in st.session_state:
    st.session_state.current_response = None
if "loading" not in st.session_state:
    st.session_state.loading = False

# 히스토리 디버깅: 쿠키에서 로드된 데이터 확인
st.write("디버깅: 쿠키에서 로드된 히스토리:", st.session_state.chat_history)

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

# 쿠키 저장 디버깅 및 데이터 최소화
def save_chat_history_to_cookies(cookies):
    try:
        # 데이터를 JSON 형식으로 변환
        history_data = json.dumps(st.session_state.chat_history)
        
        # 크기 제한 확인
        if len(history_data) > 4000:  # 4KB 제한
            st.error("히스토리 데이터가 너무 커서 쿠키에 저장할 수 없습니다.")
        else:
            # 쿠키에 저장
            cookies["chat_history"] = history_data
            cookies.save()
    except Exception as e:
        st.error(f"쿠키 저장 중 오류가 발생했습니다: {e}")

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



# 기존 코드에서 쿠키 저장 부분 수정
if st.session_state.loading and st.session_state.current_question:
    try:
        with st.spinner("답변을 생성 중입니다..."):
            # RAG 응답 생성
            retrieved_documents = retriever.invoke(st.session_state.current_question)
            response = rag_chain.invoke(st.session_state.current_question)

            # 현재 응답과 히스토리 업데이트
            documents_for_history = [
                {
                    "은행": doc.metadata.get("은행", "정보 없음"),
                    "1차분류": doc.metadata.get("1차분류", "정보 없음"),
                    "2차분류": doc.metadata.get("2차분류", "정보 없음"),
                }
                for doc in retrieved_documents
            ]

            # 응답 저장
            st.session_state.current_response = {
                "질문": st.session_state.current_question,
                "응답": response,
                # "문서": documents_for_history,
            }
            st.session_state.chat_history[-1]["응답"] = response
            # st.session_state.chat_history[-1]["문서"] = documents_for_history

            # 쿠키에 히스토리 저장 (디버깅 추가)
            save_chat_history_to_cookies(cookies)

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
