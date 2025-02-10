import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# 🔄 ChromaDB 클라이언트 생성
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 🔄 사용 모델 (384차원 출력 모델)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 차원: 384

def reset_chroma_collection():
    """기존 ChromaDB 컬렉션 삭제 후 새로 생성"""
    try:
        chroma_client.delete_collection(name="document_embeddings")
        print("🔄 기존 ChromaDB 컬렉션 삭제 완료!")
    except Exception as e:
        print(f"⚠️ 기존 컬렉션 삭제 중 오류 발생: {e}")

    collection = chroma_client.create_collection(name="document_embeddings", metadata={"dimension": 384})
    print("✅ 새로운 ChromaDB 컬렉션 생성 완료 (dim=384)!")
    return collection

def generate_embeddings(document_chunks):
    """문서 조각 리스트를 받아 임베딩 벡터 생성"""
    print(f"📌 문서 조각 수: {len(document_chunks)}")
    embeddings = embedding_model.encode(document_chunks, convert_to_numpy=True)
    print(f"✅ 임베딩 생성 완료! (차원: {embeddings.shape})")
    return embeddings

def save_embeddings_to_chromadb(embeddings, document_chunks):
    """생성된 임베딩을 ChromaDB에 저장 (384차원 고정)"""
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

def find_similar_chunks(query, top_k=3):
    """
    주어진 질문(query)에 대해 가장 유사한 문서 조각을 ChromaDB에서 검색하는 함수.
    """
    collection = chroma_client.get_collection(name="document_embeddings")
    
    # 질문(query)을 임베딩으로 변환 (query는 문자열이어야 함)
    query_embedding = embedding_model.encode([query])[0]  # query 변수 사용

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    similar_docs = []
    for idx, doc_id in enumerate(results["ids"][0]):
        doc_text = collection.get([doc_id])["metadatas"][0]["text"]
        similarity_score = results["distances"][0][idx]
        similar_docs.append((doc_text, similarity_score))

    print(f"✅ 유사한 문서 {top_k}개 검색 완료!")
    return similar_docs