readme_content = """# 비트컴퓨터_프로젝트_발표

이 프로젝트는 PDF 문서로부터 텍스트를 추출하고, 해당 문서의 내용을 기반으로 GPT 모델을 활용하여  
10개의 질문과 각 질문에 대한 구체적인 답변을 생성하는 Q&A 파이프라인을 구현한 예제입니다.  
또한, 문서의 임베딩 데이터는 ChromaDB를 통해 저장 및 관리하여, 나중에 유사 문서 검색 등 다양한 작업에 활용할 수 있습니다.

## 주요 기능

- **문서 로드 및 분할 (Loader & Splitter):**  
  PDF 파일이나 URL 등에서 문서를 읽어와 텍스트를 추출하고, 모델이 처리하기 적합한 크기로 분할합니다.

- **임베딩 생성 (Embedding):**  
  분할된 문서 조각을 SentenceTransformer 등으로 임베딩 벡터로 변환합니다.

- **ChromaDB 연동:**  
  생성된 임베딩 벡터와 메타데이터를 로컬 ChromaDB 데이터베이스에 저장합니다.  
  이 데이터베이스는 검색 시 유사 문서 조각을 빠르게 찾아내는 데 사용됩니다.

- **GPT를 활용한 질문/답변 생성 (GPT Analyzer):**  
  GPT 모델 (예: GPT-4o)을 사용하여 문서 내용을 기반으로 10개의 질문을 생성하고,  
  한 번의 호출로 10개의 질문-답변 쌍을 포함하는 JSON 배열을 생성합니다.

## 디렉토리 구조
```
project/
├── main.py               # 메인 실행 파일
├── loaders/              # 데이터 로더 모듈
│   ├── __init__.py       # 로더 모듈 초기화
│   ├── pdf_loaders.py    # PDF 로더
│   ├── hwp_loaders.py    # HWP 로더
│   ├── csv_loaders.py    # CSV/Excel 로더
│   ├── json_loaders.py   # JSON 로더
│   └── text_loaders.py   # 텍스트 로더
├── splitters/            # 텍스트 분리기 모듈
│   ├── __init__.py       # 분리기 모듈 초기화
│   └── text_splitters.py # 텍스트 분리기
├── analyzers/            # 분석 모듈
│   ├── __init__.py       # 분석기 모듈 초기화
│   └── gpt_analyzer.py   # GPT 기반 분석기
├── utils/                # 유틸리티 모듈
│   ├── __init__.py       # 유틸리티 초기화
│   ├── file_utils.py     # 파일 유형 감지 및 유틸리티 함수
│   └── env_utils.py      # 환경 변수 로드
├── requirements.txt      # 프로젝트 의존성 목록
└── README.md             # 프로젝트 설명 파일
```


## 설치 및 실행 방법

1. **환경 설정:**

   - Python 3.8 이상을 설치합니다.
   - 프로젝트 루트 디렉토리에서 의존성 패키지를 설치합니다.  
     예시:
     ```bash
     pip install -r requirements.txt
     ```
   - `.env` 파일을 생성하거나 수정하여 다음과 같이 OpenAI API 키를 설정합니다.
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   - (선택사항) 토큰 병렬 처리 관련 경고가 발생할 경우, `TOKENIZERS_PARALLELISM` 환경 변수를 설정할 수 있습니다.

2. **ChromaDB 설정:**

   - ChromaDB는 임베딩 데이터를 저장하는 로컬 데이터베이스입니다.
   - 프로젝트 실행 시 `chromadb/` 폴더가 자동 생성되며,  
     해당 폴더에는 임베딩 벡터, 메타데이터, 인덱스 파일 등이 저장됩니다.
   - 데이터베이스 파일은 텍스트 에디터로 직접 확인하기는 어렵지만,  
     ChromaDB의 Python API를 이용해 데이터를 조회할 수 있습니다.

3. **실행:**

   - 메인 파이프라인은 `main.py` 파일에 구현되어 있습니다.
   - PDF 파일 경로를 입력 데이터로 사용하며,  
     내부적으로 PDF 파일에서 텍스트 추출 → 문서 분할 → 임베딩 생성 → GPT를 통한 질문 생성 및 Q&A 생성 순으로 처리됩니다.
   - 예시:
     ```bash
     python main.py /path/to/your/document.pdf
     ```
   - 최종 결과는 10개의 질문-답변 쌍이 포함된 JSON 배열이 `Result.json` 파일에 저장됩니다.

## 주의 사항

- **LangChain Deprecation Warning:**  
  현재 코드에서는 LangChain의 `LLMChain` 클래스와 `Chain.run` 메서드를 사용합니다.  
  향후 LangChain 업데이트 시 `invoke` 메서드 등 최신 사용법으로 변경해야 할 수 있습니다.

- **Huggingface Tokenizers 경고:**  
  "The current process just got forked..."와 같은 경고는  
  병렬 처리로 인한 경고 메시지로, 큰 문제 없이 실행됩니다.  
  필요시 `TOKENIZERS_PARALLELISM` 환경 변수를 설정하여 경고를 없앨 수 있습니다.

## 참고 및 문의

- 이 프로젝트는 PDF 파일의 텍스트 추출, 문서 분할, 임베딩 생성, GPT를 이용한 질문/답변 생성 및 ChromaDB 연동을 포함한 종합적인 파이프라인을 다룹니다.
- 코드와 관련된 문의나 개선 사항은 해당 프로젝트의 GitHub 저장소 이슈 트래커를 통해 접수해 주시기 바랍니다.
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("README.md 파일이 생성되었습니다.")



