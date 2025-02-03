import os
import json
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader,
    PyPDFium2Loader,
    PDFMinerLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    TextLoader,
    WebBaseLoader,
    PythonLoader
)
from langchain_teddynote.document_loaders import HWPLoader

# USER_AGENT 설정 (HTML 로더에서 필요)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
os.environ["USER_AGENT"] = USER_AGENT

def preprocess_json(file_path):
    """
    JSON 파일에서 UTF-8 BOM 제거 처리.
    
    :param file_path: JSON 파일 경로
    :return: JSON 데이터
    """
    with open(file_path, "r", encoding="utf-8-sig") as file:
        data = json.load(file)  # JSON 파일 읽기
    return data

def load_file(file_path):
    """
    파일 경로를 기반으로 적합한 로더를 선택해 파일을 로드합니다.

    :param file_path: 파일 경로 (str)
    :return: 로드된 문서 데이터 또는 오류 메시지
    """
    # 파일 확장자 추출
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()  # 소문자로 변환하여 처리

    # 확장자에 따라 적합한 로더 선택
    if file_extension == ".pdf":
        # PDF 파일은 5가지 로더를 각각 사용
        pdf_loaders = [
            PyPDFLoader(file_path),
            PyMuPDFLoader(file_path),
            UnstructuredPDFLoader(file_path),
            PyPDFium2Loader(file_path),
            PDFMinerLoader(file_path),
        ]
        results = {}
        for loader in pdf_loaders:
            loader_name = loader.__class__.__name__  # 로더 이름 가져오기
            try:
                results[loader_name] = loader.load()  # 각 로더로 파일 로드
            except Exception as e:
                results[loader_name] = f"Error: {e}"  # 오류 발생 시 기록
        return results
    elif file_extension == ".hwp":
        loader = HWPLoader(file_path)  # HWP 로더
    elif file_extension == ".csv":
        # CSV 파일 로드
        loader = CSVLoader(file_path)
    elif file_extension in [".xls", ".xlsx"]:
        loader = UnstructuredExcelLoader(file_path)  # Excel 로더
    elif file_extension == ".txt":
        loader = TextLoader(file_path)  # TXT 파일 로더
    elif file_extension == ".json":
        # JSON 파일 사전 처리
        try:
            data = preprocess_json(file_path)  # JSON 파일 읽기
            return data
        except Exception as e:
            raise ValueError(f"JSON 파일 로드 실패: {e}")
    elif file_extension == ".html":
        # HTML 로더에서 USER_AGENT 직접 지정
        loader = WebBaseLoader(file_path, user_agent=USER_AGENT)
    elif file_extension == ".py":
        loader = PythonLoader(file_path)  # Python 파일 로더
    else:
        raise ValueError(f"지원하지 않는 파일 확장자: {file_extension}")
    
    # 다른 파일 형식 로드 및 결과 반환
    return loader.load()
