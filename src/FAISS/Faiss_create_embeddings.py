import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 데이터 로드
file_path = "data/우리은행_FAQ.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1", engine="openpyxl")

# Document 및 메타데이터 생성
documents = []
for _, row in df.iterrows():
    # 텍스트 구성
    content = f"질문: {row['질문']}\n답변: {row['응답']}"
    
    # 메타데이터 구성
    metadata = {
        "은행": row["은행"],
        "1차분류": row["1차분류"],
        "2차분류": row["2차분류"],
        "질문": row["질문"],
        "답변": row["응답"]
    }
    
    # Document 생성
    documents.append(Document(page_content=content, metadata=metadata))

# 임베딩 생성 (text-embedding-3-small 모델 사용)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents=documents, embedding=embedding_model)

# 저장
vectorstore.save_local("data/faiss_index")
print("임베딩 및 저장 완료!")
