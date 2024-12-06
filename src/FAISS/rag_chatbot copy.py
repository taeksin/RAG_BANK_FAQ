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

# 페이지 설정 (가장 첫 번째 Streamlit 명령어)
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

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "current_response" not in st.session_state:
    st.session_state.current_response = None
if "loading" not in st.session_state:
    st.session_state.loading = False

# 쿠키에서 히스토리 로드
if "loaded_history" not in st.session_state:
    chat_history = cookies.get("chat_history")
    if chat_history:
        st.session_state.chat_history = json.loads(chat_history)
    st.session_state.loaded_history = True

# 히스토리 표시
if st.session_state.chat_history:
    for i, entry in enumerate(st.session_state.chat_history):
        with st.expander(f"질문 {i+1}: {entry['질문']}"):
            st.write(f"**질문:** {entry['질문']}")
            st.write(f"**답변:** {entry['응답']}")

# 사용자 입력 처리
user_input = st.chat_input("질문을 입력하세요...")
if user_input and not st.session_state.loading:
    st.session_state.current_question = user_input
    st.session_state.loading = True  # 로딩 상태 시작
    st.session_state.current_response = None

    # 현재 질문을 히스토리에 즉시 추가 (응답은 나중에 업데이트)
    st.session_state.chat_history.append({"질문": user_input, "응답": "응답 생성 중..."})

# 응답 생성 및 상태 업데이트
if st.session_state.loading and st.session_state.current_question:
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
                "질문": doc.metadata.get("질문", "정보 없음"),
                "답변": doc.metadata.get("답변", "정보 없음"),
            }
            for doc in retrieved_documents
        ]

        # 응답 저장
        st.session_state.current_response = {
            "질문": st.session_state.current_question,
            "응답": response,
            "문서": documents_for_history,
        }
        st.session_state.chat_history[-1]["응답"] = response
        st.session_state.chat_history[-1]["문서"] = documents_for_history

        # 쿠키에 히스토리 저장
        cookies["chat_history"] = json.dumps(st.session_state.chat_history)
        cookies.save()

        # 상태 갱신
        st.session_state.loading = False
        st.session_state.current_question = None

# 현재 응답 출력
if st.session_state.loading:
    st.subheader("챗봇 응답:")
    st.write("응답을 기다리는 중입니다...")
    
    # 검색된 문서 섹션 숨기기
    st.subheader("검색된 문서:")
    st.write("응답 생성 중에는 문서를 표시할 수 없습니다.")
elif st.session_state.current_response:
    st.subheader("챗봇 응답:")
    
    # 질문과 응답을 함께 표시
    st.write(f"**질문:** {st.session_state.current_response['질문']}")
    st.write(f"**응답:** {st.session_state.current_response['응답']}")

    # 검색된 문서 출력
    st.subheader("검색된 문서:")
    with st.expander("검색된 문서 보기 (클릭하여 펼치기)"):
        for idx, doc in enumerate(st.session_state.current_response["문서"][:5], 1):  # 최대 5개 문서만 출력
            st.write(f"### 문서 {idx}:")
            st.write(f"- **은행**: {doc['은행']}")
            st.write(f"- **1차 분류**: {doc['1차분류']}")
            st.write(f"- **2차 분류**: {doc['2차분류']}")
            st.write(f"- **질문**: {doc['질문']}")
            st.write(f"- **답변**: {doc['답변']}")

            # 문서 간 구분선
            if idx < len(st.session_state.current_response["문서"][:5]):
                st.markdown("---")
