import torch
import chromadb
from transformers import AutoTokenizer, AutoModel
from config import HUGGINGFACE_API_KEY  

# ChromaDB 클라이언트 생성 (PersistentClient 사용)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# "sentence-transformers/all-MiniLM-L6-v2" 모델을 사용 (384차원 출력 모델)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", use_auth_token=HUGGINGFACE_API_KEY)
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", use_auth_token=HUGGINGFACE_API_KEY)

def mean_pooling(model_output, attention_mask):
    """
    모델 출력(token embeddings)에 대해 mean pooling을 수행합니다.
    attention_mask를 고려하여 각 토큰 임베딩의 평균을 계산합니다.
    """
    token_embeddings = model_output[0]  # 토큰 임베딩
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def generate_embeddings(document_chunks):
    """
    문서 조각 리스트(document_chunks)에 대해 Hugging Face의 transformers 모델을 사용하여 임베딩 벡터를 생성합니다.
    
    각 문서 조각에 대해:
      - 텍스트를 토크나이즈하고,
      - 모델을 통해 토큰 임베딩을 생성한 후,
      - mean pooling을 통해 하나의 임베딩 벡터로 만듭니다.
      
    반환: 각 문서 조각에 대한 임베딩 벡터(NumPy 배열) 리스트.
    """
    embeddings = []
    for text in document_chunks:
        # 텍스트 토크나이즈 (padding 및 truncation 적용)
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        # 모델 추론 (gradient 계산 불필요)
        with torch.no_grad():
            model_output = model(**encoded_input)
        # mean pooling으로 단일 임베딩 벡터 계산
        embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings.append(embedding.squeeze(0).cpu().numpy())
    return embeddings

def reset_chroma_collection():
    """
    기존 ChromaDB 컬렉션("document_embeddings")을 삭제하고 새 컬렉션을 생성합니다.
    """
    try:
        chroma_client.delete_collection(name="document_embeddings")
        print("🔄 기존 ChromaDB 컬렉션 삭제 완료!")
    except Exception as e:
        print(f"⚠️ 기존 컬렉션 삭제 중 오류 발생: {e}")
    collection = chroma_client.create_collection(name="document_embeddings", metadata={"dimension": 384})
    print("✅ 새로운 ChromaDB 컬렉션 생성 완료 (dim=384)!")
    return collection

def save_embeddings_to_chromadb(embeddings, document_chunks):
    """
    생성된 임베딩 벡터와 해당 문서 조각을 ChromaDB에 저장합니다.
    각 문서 조각에 대해 고유 ID 및 메타데이터(텍스트)를 함께 저장합니다.
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