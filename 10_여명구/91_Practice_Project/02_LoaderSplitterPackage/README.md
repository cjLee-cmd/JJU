# 파일 로더

다양한 파일 형식을 처리할 수 있는 파일 로더(file_loader.py)와 이를 실행하는 스크립트(main.py)로 구성.(모듈화)

각 파일 형식에 적합한 로더를 사용해 파일을 읽고 처리.

## 1. file_loader.py

입력받은 파일 경로(file_path)의 확장자를 분석하여 적합한 로더를 선택하고 파일을 읽음.

- PDF파일은 5가지 로더를 사용해 각각 데이터를 읽음.

- JSON은 UTF-8 BOM을 제거한 뒤 데이터를 처리.

- HTML파일은 USER_AGENT 환경 변수를 설정하여 외부 리소스를 요청할 때 브라우저 에이전트 정보를 제공해야 함.

## 2. main.py

file_loader.py의 load_file 함수를 호출해 파일 처리.


### 필요한 라이브러리 설치
pip install -q langchain_community langchain_teddynote
