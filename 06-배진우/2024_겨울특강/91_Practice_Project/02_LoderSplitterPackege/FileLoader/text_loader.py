"""
txtLoader.py
- 일반 텍스트 파일을 직접 열어서 후처리하는 로직.
"""

from langchain_community.document_loaders import TextLoader

def load_txt(file_path: str):
    """
    일반 텍스트 파일(.txt)을 열어 가공 후 반환.
    """

    # 텍스트 로더 생성
    loader = TextLoader(file_path)

    # 문서 로드
    docs = loader.load()

    return docs


if __name__ == "__main__":
    file_path = "/workspaces/JJU-1/06-배진우/2024_겨울특강/91_Practice_Project/02_LoderSplitterPackege/FileLoder/data/novel.txt"
    print(load_txt(file_path))