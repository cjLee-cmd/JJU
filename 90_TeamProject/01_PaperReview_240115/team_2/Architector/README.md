## 🌟 

이 워크플로우는 PDF 논문을 처리하고, 이를 분석하여 GPT-4o 또는 ChatPrompt API에 전달하기 위한 전체 과정입니다.
**Streamlit** 기반 인터페이스를 사용하여 사용자가 PDF를 업로드하고 자연어 쿼리를 통해 분석 결과를 얻을 수 있도록 설계되었습니다.

이 과정은 다음과 같은 주요 단계로 구성됩니다:

1. **PDF 파일 처리 및 분석** - 문서를 섹션 및 Chunk 단위로 나누어 세그멘테이션 수행.
2. **벡터 데이터베이스 생성** - 추출된 텍스트를 벡터화하여 검색 가능한 데이터베이스 생성.
3. **질의 응답 생성** - 사용자의 자연어 쿼리에 기반한 적합한 응답 생성.
4. **Streamlit 인터페이스 제공** - 사용자 친화적 인터페이스로 기능을 제공.

---

## 🛠️ 설치 및 실행 방법

### 1️⃣ 요구 사항
- Python 3.x 이상
- 필요한 라이브러리 설치 (아래 참조)

#### 필요한 라이브러리 (`requirements.txt`)
```
langchain-chroma
langchain-openai
langchain-teddynote
langchain-community
langchain-experimental
streamlit
pymupdf
python-dotenv
chromadb
```

## 📝 워크플로우 상세 설명

### 1️⃣ PDF 파일 처리 및 분석
- **목적**: PDF 파일을 섹션 및 의미 단위(Chunk)로 분할하여 분석 준비.
- **구현**: `process_pdf` 함수에서 PyMuPDF와 SemanticChunker를 사용하여 텍스트를 분할.
- **주요 흐름**:
  - PDF 파일 로드 (`PyMuPDFLoader`)
  - 페이지 단위 텍스트 추출
  - 의미 단위로 Chunk 생성 (`SemanticChunker`)

### 2️⃣ 벡터 데이터베이스 생성
- **목적**: Chunk로 나눈 텍스트를 벡터화하여 검색 가능한 데이터베이스를 구축.
- **구현**: `create_vector_database` 함수에서 LangChain의 Chroma 모듈 활용.
- **주요 흐름**:
  - Embedding 모델 초기화 (`OpenAIEmbeddings`)
  - Chroma 데이터베이스 생성 및 데이터 저장

### 3️⃣ 데이터베이스 쿼리 및 질의 응답 생성
- **목적**: 사용자의 질문에 대해 데이터베이스에서 최적의 정보를 검색하고 GPT 모델을 통해 응답 생성.
- **구현**:
  - `query_database` 함수: MMR(Maximal Marginal Relevance) 검색 수행
  - `generate_response` 함수: 검색된 문서와 사용자 질문을 GPT 모델에 전달
- **주요 흐름**:
  - 벡터 데이터베이스 검색 → 관련 문서 추출 → GPT 프롬프트 생성 → 응답 생성

### 4️⃣ Streamlit 인터페이스 제공
- **목적**: 사용자 친화적인 웹 인터페이스 제공.
- **구현**: `main_streamlit` 함수에서 Streamlit을 사용하여 파일 업로드, 쿼리 입력, 결과 표시.
- **주요 기능**:
  - PDF 업로드 위젯
  - 자연어 쿼리 입력 필드
  - 결과 표시 및 응답 애니메이션 (`stream_response` 사용)

---

## 🔗 주요 함수 흐름 및 연결

1. **PDF 분석 및 데이터베이스 생성**:
   - `process_pdf` → `create_vector_database`

2. **질문 응답 흐름**:
   - `query_database` → `generate_response`

3. **Streamlit 인터페이스**:
   - 파일 업로드 → 쿼리 입력 → 분석 결과 표시

---

## 📋 참고사항
- **Streamlit**: 이 애플리케이션은 Streamlit 기반으로 구축되었으며, 실행 시 로컬 브라우저에서 확인 가능합니다.
- **OpenAI API**: `.env` 파일에 OpenAI API 키를 저장하여 사용해야 합니다.
- **데이터베이스 저장**: Chroma 데이터베이스는 `./db/chromadb` 디렉터리에 저장됩니다.

---
