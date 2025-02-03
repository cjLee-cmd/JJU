# BM25와 FAISS 기반 앙상블 검색 시스템

## 개요
본 프로젝트는 BM25와 FAISS 기반의 검색 시스템을 구축하고, 이를 `EnsembleRetriever`로 통합하여 보다 정교한 검색 결과를 제공하는 코드입니다. OpenAI 임베딩을 활용해 문서를 벡터화하고, 서로 다른 검색 알고리즘의 장점을 결합하여 최적의 결과를 도출합니다.

---

## 각 함수에 대한 설명

### 1. `initialize_retrievers(doc_list, k=1)`
- **설명**: 입력된 문서 리스트를 기반으로 BM25 및 FAISS 리트리버를 초기화하고, 이를 `EnsembleRetriever`로 통합합니다.
- **매개변수**:
  - `doc_list` (list): 검색 대상으로 사용할 문서의 리스트.
  - `k` (int): 검색 결과로 반환할 상위 문서의 개수 (기본값: 1).
- **반환값**: 구성된 `EnsembleRetriever` 객체.

### 2. `retrieve_documents(query, retriever, config)`
- **설명**: 주어진 쿼리를 기반으로 설정된 리트리버와 구성(config)을 사용해 검색 결과를 반환합니다.
- **매개변수**:
  - `query` (str): 사용자가 입력한 검색 쿼리.
  - `retriever` (EnsembleRetriever): 초기화된 앙상블 리트리버 객체.
  - `config` (dict): 리트리버의 가중치와 검색 매개변수 설정.
- **반환값**: 검색된 문서 리스트.

---

## 코드 동작 원리
1. **BM25 리트리버 초기화**:
   - 입력 문서 리스트에서 단어 빈도 기반의 BM25 알고리즘을 활용해 검색 결과를 도출.
2. **FAISS 리트리버 초기화**:
   - OpenAI 임베딩을 통해 문서를 벡터화한 후, FAISS 라이브러리를 사용해 벡터 간 유사도를 기반으로 검색 결과를 반환.
3. **EnsembleRetriever 통합**:
   - BM25와 FAISS의 검색 결과를 `EnsembleRetriever`로 결합하여 가중치 기반의 최종 결과 생성.
4. **사용자 쿼리 처리**:
   - 사용자가 입력한 검색 쿼리를 바탕으로 초기화된 리트리버와 설정(config)을 사용해 검색 결과 반환.

---

## 코드 동작 방법
1. 필요한 라이브러리를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```

2. `main` 함수가 포함된 코드를 실행합니다:
   ```bash
   python 10-03_EnsembleRetriever.py
   ```
3. 검색 결과는 콘솔에 출력됩니다.

---

## 라이브러리에 대한 설명

### 1. `langchain`
- **기능**: 문서 검색 및 처리에 필요한 다양한 리트리버와 모델을 지원하는 라이브러리.
- **사용 목적**: BM25 및 FAISS 리트리버를 쉽게 초기화하고, 이를 결합하여 앙상블 검색 시스템 구축.

### 2. `faiss-cpu`
- **기능**: 대규모 벡터 데이터의 효율적인 검색을 위한 라이브러리.
- **사용 목적**: OpenAI 임베딩으로 변환된 문서 벡터를 효율적으로 검색.

### 3. `langchain-openai`
- **기능**: OpenAI의 임베딩 기능과 통합하여 텍스트 데이터를 벡터화.
- **사용 목적**: 문서를 벡터화하여 FAISS 리트리버의 입력 데이터로 사용.

---

## 전체 코드에 대한 설명
1. **문서 초기화**:
   - 검색 대상으로 사용할 문서를 리스트 형식으로 준비합니다.
2. **리트리버 초기화**:
   - `initialize_retrievers` 함수에서 BM25와 FAISS 리트리버를 생성하고, `EnsembleRetriever`로 결합합니다.
3. **검색 실행**:
   - `retrieve_documents` 함수에서 사용자가 입력한 쿼리를 처리하고, 설정된 가중치에 따라 앙상블 결과를 반환합니다.
4. **출력**:
   - 최종 검색 결과를 콘솔에 출력하여 사용자에게 제공합니다.

---

## 예시
- 샘플 문서 리스트:
  ```python
  doc_list = [
      "I like apples",
      "I like apple company",
      "I like apple's iphone",
      "Apple is my favorite company",
      "I like apple's ipad",
      "I like apple's macbook",
  ]
  ```
- 사용자가 입력한 쿼리: `"my favorite fruit is apple"`
- 결과:
  - BM25와 FAISS 리트리버의 검색 결과를 가중치(0.7:0.3)에 따라 결합한 최적의 검색 결과를 반환.