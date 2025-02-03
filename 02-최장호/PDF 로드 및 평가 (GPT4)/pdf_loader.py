import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, PyPDFium2Loader, PDFMinerLoader, PDFPlumberLoader

class PDFLoader:
    def __init__(self, file_path: str):
        """
        PDF 파일에서 텍스트를 추출하기 위한 로더 클래스
        :param file_path: PDF 파일 경로
        """
        self.file_path = file_path
        self.loaders = [
            PyPDFLoader(self.file_path),
            PyMuPDFLoader(self.file_path),
            PyPDFium2Loader(self.file_path),
            PDFMinerLoader(self.file_path),
            PDFPlumberLoader(self.file_path)
        ]
    
    def extract_texts(self):
        """
        PDF 파일에서 텍스트를 추출하여 리스트에 저장
        """
        extracted_texts = []
        for loader in self.loaders:
            try:
                docs = loader.load()
                text = docs[0].page_content[:500]  # 첫 500자만 추출
                extracted_texts.append(f"Loader: {loader.__class__.__name__}\n{text}")
            except Exception as e:
                extracted_texts.append(f"Loader: {loader.__class__.__name__} encountered an error: {e}")
        return extracted_texts
