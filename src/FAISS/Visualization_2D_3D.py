import platform
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.decomposition import PCA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

DIMENSION = 3

# 한글 폰트 설정
if platform.system() == "Windows":
    rc('font', family='Malgun Gothic')  # 윈도우: 맑은 고딕
elif platform.system() == "Darwin":
    rc('font', family='AppleGothic')   # 맥OS: 애플고딕
else:
    rc('font', family='NanumGothic')   # 리눅스: 나눔고딕

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 폰트 깨짐 방지

# FAISS 데이터 로드
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local("data/faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

# 벡터와 메타데이터 추출
vectors = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)  # 모든 벡터 추출

# 2D 시각화 (PCA 사용, 좌표를 라벨로 사용)
def plot_2d_with_pca(vectors):
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)
    
    # 라벨로 좌표 사용
    labels = [f"({x:.2f}, {y:.2f})" for x, y in reduced_vectors]

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], s=10, c='blue')
    
    # 각 점에 라벨 추가
    for i, label in enumerate(labels):
        plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], label, fontsize=8, alpha=0.7)
    
    plt.title("2D Visualization of Vectors (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# 3D 시각화 (PCA 사용, 좌표를 라벨로 사용)
def plot_3d_with_pca(vectors):
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)
    
    # 라벨로 좌표 사용
    labels = [f"({x:.2f}, {y:.2f}, {z:.2f})" for x, y, z in reduced_vectors]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], s=10, c='green')
    
    # 각 점에 라벨 추가
    for i, label in enumerate(labels):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], label, fontsize=8, alpha=0.7)
    
    ax.set_title("3D Visualization of Vectors (PCA)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    plt.show()

# 실행
if DIMENSION == 2:
    plot_2d_with_pca(vectors)
else:
    plot_3d_with_pca(vectors)
