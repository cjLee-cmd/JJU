"""
hwpLoader.py
- 한글(HWP) 파일을 직접 열어서 파싱하는 로직.
- pyhwp, olefile 등을 사용할 수 있습니다.
"""

from langchain_teddynote.document_loaders import HWPLoader

def load_hwp(file_path: str):
    """
    HWP 파일을 바이너리 모드로 열어 필요한 정보 추출 후 반환.
    """

    # HWP Loader 객체 생성
    loader = HWPLoader(file_path)

    # 문서 로드
    docs = loader.load()

    return docs

if __name__ == "__main__":
    file_path = "/workspaces/JJU-1/06-배진우/2024_겨울특강/91_Practice_Project/02_LoderSplitterPackege/FileLoder/data/2025년도 제48회 보험계리사 및 손해사정사 시험 시행계획 공고_최종.hwp"
    print(load_hwp(file_path))