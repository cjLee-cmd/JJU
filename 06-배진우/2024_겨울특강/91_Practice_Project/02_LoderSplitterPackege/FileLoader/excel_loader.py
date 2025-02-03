"""
execlLoader.py
- 엑셀 파일을 직접 열어서 파싱하는 로직.
"""
from langchain_community.document_loaders import UnstructuredExcelLoader

def load_excel(file_path: str):
    # UnstructuredExcelLoader 생성
    loader = UnstructuredExcelLoader(file_path, mode="elements")

    # 문서 로드
    docs = loader.load()

    return docs

if __name__ == "__main__":
    file_path = "/workspaces/JJU-1/06-배진우/2024_겨울특강/91_Practice_Project/02_LoderSplitterPackege/FileLoder/data/서울2019년기상정보.xlsx"
    print(load_excel(file_path))