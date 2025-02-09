# README.md
# 프로젝트 이름: 데이터 분석 및 처리 파이프라인

## 프로젝트 설명
이 프로젝트는 다양한 유형의 데이터를 로드하고, 텍스트를 분리하며, OpenAI의 GPT 모델을 이용해 분석하는 파이프라인을 제공합니다. 데이터를 효율적으로 처리하고 분석 결과를 도출하기 위해 모듈화된 구조로 설계되었습니다.

---

## 기능
1. **데이터 로딩**
   - PDF, HWP, CSV, Excel, JSON, 텍스트 파일 및 URL 데이터를 지원합니다.
   - 파일 유형에 따라 적합한 로더를 선택하여 데이터를 로드합니다.

2. **텍스트 분리**
   - 다양한 텍스트 분리기(Text Splitter)를 사용하여 데이터를 조각으로 나눕니다.
   - 파일 유형별로 적합한 텍스트 분리기를 자동으로 매핑합니다.

3. **데이터 분석**
   - OpenAI GPT-4 모델을 활용하여 데이터의 주요 포인트를 요약하고 품질을 평가합니다.
   - 데이터 분석 결과를 상세히 출력합니다.

4. **환경 변수 관리**
   - `.env` 파일을 이용해 API 키와 사용자 에이전트를 안전하게 관리합니다.

---

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

---

## 요구 사항
이 프로젝트를 실행하려면 다음이 필요합니다:

- Python 3.8 이상
- OpenAI API 키

필요한 라이브러리는 `requirements.txt`를 사용해 설치할 수 있습니다:
```bash
pip install -r requirements.txt
```

---

## 실행 방법
1. 환경 변수 설정
   `.env` 파일을 생성하고 다음 내용을 추가합니다:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   USER_AGENT=your_user_agent
   ```

2. 스크립트 실행
   데이터를 로드하고 분석하려면 `main.py`를 실행합니다:
   ```bash
   python main.py
   ```

3. 입력 데이터 설정
   `main.py`의 `input_data` 변수에 분석할 파일 경로나 URL을 지정합니다.

---

## 라이선스
이 프로젝트는 [MIT 라이선스](LICENSE)를 따릅니다.

## 기여
기여를 환영합니다! 개선 사항이나 버그를 발견하면 이슈를 생성하거나 풀 리퀘스트를 제출해주세요.