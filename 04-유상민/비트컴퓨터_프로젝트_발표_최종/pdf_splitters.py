# pdf_splitters.py
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

def get_pdf_splitters():
    """
    PDF 파일에 사용할 스플리터들을 딕셔너리 형태로 반환합니다.
    각 스플리터는 chunk_size=100, chunk_overlap=0으로 설정됩니다.
    """
    splitters = {
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
        "CharacterTextSplitter": CharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
        "TokenTextSplitter": TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    }
    return splitters

def split_content(content: str) -> dict:
    """
    전달받은 content(텍스트)를 각 스플리터로 분할하여 그 결과를 딕셔너리로 반환합니다.
    반환 형식: {스플리터 이름: 청크 리스트}
    """
    splitters = get_pdf_splitters()
    split_results = {}
    for name, splitter in splitters.items():
        try:
            chunks = splitter.split_text(content)
            split_results[name] = chunks
        except Exception as e:
            split_results[name] = f"Error: {e}"
    return split_results