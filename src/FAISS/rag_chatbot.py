import os
import streamlit as st
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

# Streamlit 기본 설정
st.set_page_config(page_title="RAG 기반 챗봇", layout="wide")
st.title("RAG 기반 FAQ 챗봇 🤖")

# 임베딩 모델 생성
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 현재 파일 기준으로 두 단계 상위 폴더로 이동 후 FAISS 저장소 경로 설정
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

# 질문 입력 및 엔터키로 제출
question = st.text_input("질문을 입력하세요:", placeholder="예: 인터넷/스마트뱅킹 이체한도 조회 및 증액하는 방법 알려줘", on_change=lambda: st.session_state.submit_question(), key="question_input")

# 세션 상태로 버튼 동작 관리
if "submit_question" not in st.session_state:
    st.session_state.submit_question = lambda: None

# "질문하기" 버튼 동작
if st.button("질문하기") or st.session_state.submit_question:
    st.session_state.submit_question = lambda: None  # 초기화
    if question.strip():
        with st.spinner("답변을 생성 중입니다..."):
            # 질문을 임베딩
            question_embedding = embedding_model.embed_query(question)

            # 리트리버에서 문서 검색
            retrieved_documents = vectorstore.similarity_search_by_vector(question_embedding, k=5)

            # RAG를 사용하여 응답 생성
            response = rag_chain.invoke(question)

            # 응답 출력
            st.subheader("챗봇 응답:")
            st.write(response)

            # 검색된 문서 출력 (전체를 하나의 토글로 표시)
            st.subheader("검색된 문서:")
            with st.expander("검색된 문서 보기 (클릭하여 펼치기)"):

                for idx, doc in enumerate(retrieved_documents[:5], 1):  # 최대 5개 문서만 출력
                    st.write(f"### 문서 {idx}:")
                    st.write(f"- **은행**: {doc.metadata.get('은행', '정보 없음')}")
                    st.write(f"- **1차 분류**: {doc.metadata.get('1차분류', '정보 없음')}")
                    st.write(f"- **2차 분류**: {doc.metadata.get('2차분류', '정보 없음')}")
                    st.write(f"- **질문**: {doc.metadata.get('질문', '정보 없음')}")
                    st.write(f"- **답변**: {doc.metadata.get('답변', '정보 없음')}")

                    # 문서 간 구분선
                    if idx < len(retrieved_documents[:5]):  # 마지막 문서 이후에는 구분선 추가하지 않음
                        st.markdown("---")  # 구분선
    else:
        st.warning("질문을 입력해주세요.")
