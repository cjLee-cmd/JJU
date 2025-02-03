"""
PDFLoader.py
- PDF 파일을 직접 열어서 파싱하는 로직.
"""

from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    PyPDFium2Loader,
    PDFMinerLoader,
    PDFPlumberLoader
)

class PDFLoader:
    @staticmethod
    def load_PyPDF(file_path: str):
        # load_PyPDF 로더 인스턴스 생성
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return docs

    @staticmethod
    def load_PyMuPDF(file_path: str):
        # PyMuPDF 로더 인스턴스 생성
        loader = PyMuPDFLoader(file_path)
        docs = loader.load() # 문서 로드
        return docs
    
    @staticmethod
    def load_PyPDFium2(file_path: str):
        # PyPDFium2 로더 인스턴스 생성
        loader = PyPDFium2Loader(file_path)
        docs = loader.load()
        return docs

    @staticmethod
    def load_PDFMiner(file_path: str):
         # PyMuPDF 로더 인스턴스 생성
        loader = PDFMinerLoader(file_path)
        docs = loader.load() # 문서 로드       
        return docs
    
    @staticmethod
    def load_PDFPlumber(file_path: str):
        # PDF 문서 로더 인스턴스 생성
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        return docs


if __name__ == "__main__":
    file_path = "/workspaces/JJU-1/06-배진우/2024_겨울특강/91_Practice_Project/02_LoderSplitterPackege/FileLoder/data/TransUNet.pdf"
    print(PDFLoader.load_PyPDF(file_path))