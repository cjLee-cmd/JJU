"""
pdf 문서 pdf_loader.py 입니다.

- 여러가지 LangChain에 pdf Loader들을 함수로 정의한 파일
"""

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, PyPDFium2Loader, PDFMinerLoader, PDFPlumberLoader

def py_pdf_loader(FILE_PATH, OCR=True):
    # 파일 경로 설정
    loader = PyPDFLoader(FILE_PATH, extract_images=OCR)

    # PDF 로더 초기화
    docs = loader.load()

    # 문서의 내용 반환
    return docs

def py_mu_pdf(FILE_PATH):
    # PyMuPDF 로더 인스턴스 생성
    loader = PyMuPDFLoader(FILE_PATH)

    # 문서 로드
    docs = loader.load()

    # 문서의 내용 반환
    return docs

def py_pdf_mu2loader(FILE_PATH):
    # PyPDFium2 로더 인스턴스 생성
    loader = PyPDFium2Loader(FILE_PATH)

    # 데이터 로드
    docs = loader.load()

    # 문서의 내용 반환
    return docs

def pdf_minerloader(FILE_PATH):
    # PDFMiner 로더 인스턴스 생성
    loader = PDFMinerLoader(FILE_PATH)

    # 데이터 로드
    docs = loader.load()

    # 문서의 내용 반환
    return docs

def pdf_plumber(FILE_PATH):
    # PDF 문서 로더 인스턴스 생성
    loader = PDFPlumberLoader(FILE_PATH)

    # 문서 로딩
    docs = loader.load()

    # 문서의 내용 반환
    return docs