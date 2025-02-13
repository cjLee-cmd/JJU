# PDF 평가 및 QA 워크플로우 프로젝트

이 프로젝트는 PDF 파일을 업로드하여 여러 로더와 스플리터를 통해 문서의 텍스트를 추출 및 분할하고, GPT 기반 평가를 통해 최적의 로더/스플리터 조합을 선택합니다. 선택된 조합의 텍스트는 임베딩되어 ChromaDB에 저장되며, 저장된 데이터를 바탕으로 문서와 관련된 질문 20개를 생성하고, 각 질문에 대한 답변 및 평가를 수행합니다. 최종적으로 QA 평가 결과를 JSON 파일로 다운로드할 수 있습니다.

## 주요 기능

- **PDF 텍스트 추출 및 분할**  
  - 여러 종류의 PDF 로더 (PDF, MuPDF, PDFium, PDFMiner, PDFPlumber)를 사용하여 문서 전체의 텍스트를 추출합니다.
  - 텍스트를 여러 스플리터(Recursive, Character, Token 등)로 분할하여 평가합니다.
  
- **GPT 기반 평가**  
  - 각 로더/스플리터 조합에 대해 GPT 모델을 활용하여 문서 구조, 문맥 완성도, 청크 크기, 연속성, 가독성 등을 평가합니다.
  - 평가 결과를 바탕으로 최고 평가 조합을 선택합니다.

- **문서 임베딩 및 ChromaDB 저장**  
  - 최고 평가 조합의 텍스트를 SentenceTransformer(`all-MiniLM-L6-v2`)를 사용해 임베딩하고, ChromaDB에 저장합니다.

- **QA 평가 파이프라인**  
  - 저장된 문서를 바탕으로 20개의 관련 질문을 자동 생성합니다.
  - 각 질문에 대해 문서 조각을 검색하여 답변을 생성하고, GPT 기반 평가로 답변의 정확성, 관련성, 완전성, 명료성을 평가합니다.
  - QA 평가 결과는 JSON 파일로 다운로드할 수 있어, 다른 모델의 결과와 비교할 수 있습니다.

- **Streamlit 기반 UI**  
  - 사용자는 웹 인터페이스에서 PDF 파일을 업로드하고, 평가 결과와 QA 평가 과정을 실시간으로 확인할 수 있습니다.
  - 우측 하단에는 깔끔한 모던 아이콘(개발자 정보, 모델 정보)이 제공됩니다.

## 프로젝트 구조

├── config.py             # API 키, 모델 및 기타 기본 설정
├── pdf_loaders.py        # PDF 파일을 다양한 로더를 통해읽어들이는 모듈
├── pdf_splitters.py      # 추출된 텍스트를 다양한 스플리터로 분할하는 모듈
├── evaluate.py           # GPT를 활용한 로더/스플리터 평가 모듈
├── embed_best.py         # 최고 평가 조합의 텍스트를 임베딩하고 ChromaDB에 저장하는 모듈
├── qa_evaluation.py      # 문서 기반 QA 질문 생성, 답변 및 평가 파이프라인 모듈
└── main.py               # Streamlit 기반 UI 및 전체 워크플로우 통합 실행 파일

## 설치 및 실행

1. **필수 패키지 설치**  
   아래와 같이 pip를 사용하여 필요한 패키지를 설치합니다.
   ```bash
   pip install streamlit pandas chromadb sentence-transformers python-dotenv langchain-community

2.	환경 변수 설정
프로젝트 루트 디렉토리에 .env 파일을 생성하고, OpenAI API 키를 설정합니다.
OPENAI_API_KEY=your_openai_api_key_here

3.	프로젝트 실행
Streamlit 앱을 실행합니다.
streamlit run main.py


## 사용 방법
1.	PDF 파일 업로드
웹 인터페이스에서 PDF 파일을 업로드합니다.
2.	문서 평가
업로드된 파일에 대해 여러 로더와 스플리터를 적용하여 텍스트를 추출 및 분할하고, GPT를 통한 평가 결과를 확인합니다.
3.	최고 평가 조합 임베딩
최고 평가 조합의 텍스트가 임베딩되어 ChromaDB에 저장됩니다.
4.	QA 평가 파이프라인
저장된 문서를 기반으로 자동 생성된 20개의 질문에 대해, 관련 답변 생성 및 평가가 진행됩니다.
•	QA 평가 결과는 웹 화면에 표시되며, JSON 파일로 다운로드할 수 있습니다.
5.	결과 비교
다운로드한 JSON 파일을 사용하여 다른 모델의 QA 평가 결과와 비교할 수 있습니다.

개발자 정보 및 모델
- 개발자: 유상민
- 사용 모델: GPT 모델 (예: gpt-4o-mini), 임베딩 모델: all-MiniLM-L6-v2

참고
- 이 프로젝트는 OpenAI의 GPT 모델과 LangChain, ChromaDB, SentenceTransformer 등을 활용하여 구현되었습니다.
- 프로젝트의 각 모듈은 독립적으로 동작하며, 전체 파이프라인은 Streamlit을 통해 통합 실행됩니

