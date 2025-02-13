# embed_best.py
import chromadb
import numpy
from sentence_transformers import SentenceTransformer

# ChromaDB 클라이언트 생성 (PersistentClient 사용)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 사용 모델: all-MiniLM-L6-v2 (384차원 출력)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def reset_chroma_collection():
    """
    기존 ChromaDB 컬렉션을 삭제하고 새 컬렉션을 생성합니다.
    """
    try:
        chroma_client.delete_collection(name="document_embeddings")
        print("🔄 기존 ChromaDB 컬렉션 삭제 완료!")
    except Exception as e:
        print(f"⚠️ 기존 컬렉션 삭제 중 오류 발생: {e}")
    collection = chroma_client.create_collection(name="document_embeddings", metadata={"dimension": 384})
    print("✅ 새로운 ChromaDB 컬렉션 생성 완료 (dim=384)!")
    return collection

def generate_embeddings(document_chunks):
    """
    문서 조각 리스트를 받아 임베딩 벡터를 생성합니다.
    
    반환:
        numpy 배열 형식의 임베딩 (shape 정보 출력)
    """
    print(f"📌 문서 조각 수: {len(document_chunks)}")
    embeddings = embedding_model.encode(document_chunks, convert_to_numpy=True)
    print(f"✅ 임베딩 생성 완료! (차원: {embeddings.shape})")
    return embeddings

def save_embeddings_to_chromadb(embeddings, document_chunks):
    """
    생성된 임베딩 벡터와 해당 문서 조각을 ChromaDB에 저장합니다.
    """
    collection = reset_chroma_collection()
    for i, (embedding, text) in enumerate(zip(embeddings, document_chunks)):
        if len(embedding) != 384:
            print(f"⚠️ 경고: 임베딩 벡터 차원이 {len(embedding)}입니다. 384로 변환 필요!")
            continue
        collection.add(
            ids=[f"doc_{i}"],
            embeddings=[embedding.tolist()],
            metadatas=[{"text": text}]
        )
    print("✅ ChromaDB에 임베딩 저장 완료!")

# -------------------------------------------------------------------
# 여기서 best_result_text에는 최종 평가 결과에서 선택한 최고 조합의 전체 텍스트가 들어있어야 합니다.
# 예시로, 아래와 같이 변수에 값을 할당합니다.
# 실제 시스템에서는 평가 파이프라인에서 "최고 평가 조합"을 선택한 후 해당 텍스트를 전달하면 됩니다.
best_result_text = "여기에 최고의 평가 결과 조합에 해당하는 전체 텍스트가 들어갑니다."

# 문서 조각 리스트에 단일 텍스트(최고 결과)를 담아서 임베딩 진행
document_chunks = [best_result_text]

# 임베딩 생성
embeddings = generate_embeddings(document_chunks)

# 생성된 임베딩을 ChromaDB에 저장
save_embeddings_to_chromadb(embeddings, document_chunks)