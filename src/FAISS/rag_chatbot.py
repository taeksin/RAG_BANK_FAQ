import os
import json
import time
import streamlit as st
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# 페이지 설정
st.set_page_config(page_title="FAQ 챗봇", layout="wide")

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# Streamlit 기본 설정
st.title("RAG 기반 FAQ 챗봇 🤖")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

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
    return {}  # 히스토리가 없으면 빈 딕셔너리 반환

# 히스토리 저장 함수
def save_chat_history():
    try:
        with open(history_file_path, "w", encoding="utf-8") as file:
            json.dump(st.session_state.chat_history, file, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"히스토리 저장 중 오류가 발생했습니다: {e}")

# UID 세션 상태 초기화 및 저장
if "uid" not in st.session_state:
    st.session_state.uid = str(int(time.time()))  # 새 UID 생성 (현재 시간 기반)

# # 세션 상태에 저장된 UID 표시
# st.write(f"사용자의 UID: {st.session_state.uid}")

# 히스토리 파일 로드
st.session_state.chat_history = load_chat_history()

# 임베딩 모델 생성
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# FAISS 저장소 경로 설정
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
faiss_path = os.path.join(base_dir, "data/faiss_index_clean")

# 벡터 저장소 로드
vectorstore = FAISS.load_local(faiss_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 10, 'lambda_mult': 0.9})

# 프롬프트 파일 경로 설정
prompt_file_path = "src/FAISS/prompt.txt"

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

# 히스토리 표시 함수
def display_history(container):
    with container:
        if st.session_state.chat_history:
            st.subheader("히스토리:")

            # 현재 사용자 uid에 해당하는 히스토리만 표시
            user_history = st.session_state.chat_history.get(st.session_state.uid, [])

            if user_history:  # 해당 사용자의 히스토리가 있을 경우
                for entry in user_history:
                    # 질문 말풍선
                    st.markdown(
                        f"""
                        <div style="text-align: right; margin-bottom: 10px;">
                            <div style="
                                display: inline-block;
                                background-color: #DCF8C6;
                                border-radius: 10px;
                                padding: 10px 15px;
                                color: black;
                                max-width: 80%;
                                word-wrap: break-word;
                                box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
                            ">
                                {entry['질문']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True
                    )

                    # 응답 말풍선
                    st.markdown(
                        f"""
                        <div style="text-align: left; margin-bottom: 10px;">
                            <div style="
                                display: inline-block;
                                background-color: #FFFFFF;
                                border-radius: 10px;
                                padding: 10px 15px;
                                color: black;
                                max-width: 95%;
                                word-wrap: break-word;
                                box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
                                border: 1px solid #E6E6E6;
                            ">
                                {entry['응답']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True
                    )
            else:
                st.write("현재 히스토리가 없습니다.")

# UI 컨테이너 생성
history_container = st.container()

# 사용자 입력 처리
user_input = st.chat_input("질문을 입력하세요...")
if user_input:
    # 입력값 유효성 검사
    if not st.session_state.loading:
        st.session_state.current_question = user_input
        st.session_state.loading = True  # 로딩 상태 시작
        st.session_state.current_response = None

        # 현재 질문을 히스토리에 즉시 추가 (응답은 나중에 업데이트)
        if st.session_state.uid not in st.session_state.chat_history:
            st.session_state.chat_history[st.session_state.uid] = []

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
            st.session_state.chat_history[st.session_state.uid].append({
                "질문": st.session_state.current_question,
                "응답": response
            })

            # 히스토리 저장
            save_chat_history()

    except Exception as e:
        st.error(f"응답 생성 중 오류가 발생했습니다: {e}")
        st.session_state.chat_history[st.session_state.uid].append({
            "질문": st.session_state.current_question,
            "응답": "응답 생성에 실패했습니다. 다시 시도해주세요."
        })
    finally:
        # 상태 갱신
        st.session_state.loading = False
        st.session_state.current_question = None

        # 응답 생성 후 히스토리 컨테이너를 새로 렌더링
        history_container.empty()  # 기존 내용을 지움
        display_history(history_container)  # 새로 렌더링

# # 현재 응답 출력
# if st.session_state.loading:
#     st.subheader("챗봇 응답:")
#     st.write("응답을 기다리는 중입니다...")

# elif st.session_state.current_response:
#     st.subheader("챗봇 응답:")
#     st.write(f"**질문:** {st.session_state.current_response['질문']}")
#     st.write(f"**응답:** {st.session_state.current_response['응답']}")
