"""
FileLoader 패키지 초기화 파일(__init__.py)
- 외부에서 패키지를 임포트할 때, 주요 로더 기능들을 간편하게 사용할 수 있도록 구성합니다.
"""

# PDFLoader 클래스 임포트
from .pdf_loader import PDFLoader

# 다른 로더 모듈에서 필요한 함수들을 임포트
from .json_loader import load_json
from .csv_loader import load_csv
from .excel_loader import load_excel
from .hwp_loader import load_hwp
from .text_loader import load_txt

# __all__을 사용하여 패키지에서 공개할 항목을 명시
__all__ = [
    "PDFLoader",
    "load_txt",
    "load_json",
    "load_csv",
    "load_excel",
    "load_hwp",
]