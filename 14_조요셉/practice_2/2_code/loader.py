"""
문서 loader.py 입니다.

- 여러가지 LangChain에 Loader들을 함수로 정의한 파일
"""

from langchain_teddynote.document_loaders import HWPLoader
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader, TextLoader, JSONLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders.csv_loader import CSVLoader, UnstructuredCSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders.python import PythonLoader

# 한글 문서 로드
def hwp_loader(FILE_PATH):
    # HWP Loader 객체 생성
    loader = HWPLoader(FILE_PATH)

    # loader 반환
    return loader

# csv 파일 로드
def csv_loader(FILE_PATH):
    # CSV 로더 생성
    loader = CSVLoader(file_path=FILE_PATH)

    # loader 반환
    return loader

def unstructured_csv_loader(FILE_PATH):
    # 비구조화 CSV 로더 인스턴스 생성
    loader = UnstructuredCSVLoader(file_path=FILE_PATH, mode="elements")

    # loader 반환
    return loader

# Excel 파일 로드
def excel_loader(FILE_PATH):
    # UnstructuredExcelLoader 생성
    loader = UnstructuredExcelLoader(file_path=FILE_PATH, mode="elements")

    # loader 반환
    return loader

# Word 파일 로드
def word_loader(FILE_PATH):
    # Docx2txtLoader 생성
    loader = Docx2txtLoader(FILE_PATH)

    # loader 반환
    return loader

def unstructured_word_loader(FILE_PATH):
    # 비구조화된 워드 문서 로더 생성
    loader = UnstructuredWordDocumentLoader(FILE_PATH)

    # loader 반환
    return loader

# txt 파일 로드
def txt_loader(FILE_PATH):
    # 텍스트 로더 생성
    loader = TextLoader(FILE_PATH)

    # loader 반환
    return loader

# json 파일 로드
def json_loader(FILE_PATH):
    # JSONLoader 생성
    loader = JSONLoader(
        file_path=FILE_PATH,
        jq_schema=".[].phoneNumbers",
        text_content=False,
    )

    # loader 반환
    return loader

# py 파일 로드
def python_loader(FILE_PATH):
    # PythonLoader 생성
    loader = PythonLoader(FILE_PATH)

    # loader 반환
    return loader

# html 파일 로드
def html_loader(FILE_PATH):
    # UnstructuredHTMLLoader 생성
    loader = UnstructuredHTMLLoader(FILE_PATH)

    # loader 반환
    return loader

# markdown 파일 로드
def markdown_loader(FILE_PATH):
    # UnstructuredMarkdownLoader 생성
    loader = UnstructuredMarkdownLoader(FILE_PATH)

    # loader 반환
    return loader

def main(loader):

    docs = loader.load()

    return docs