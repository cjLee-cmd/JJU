from pdf_processor import PDFProcessor
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

# .env 파일에서 OpenAI API 키 가져오기
api_key = os.getenv('OPENAI_API_KEY')

# PDF 파일 경로 설정 (예: 파일 경로를 원하는 위치로 수정하세요)
file_path = "C:/Users/kowm6/Desktop/소상공인 정책 자금.pdf"

# PDFProcessor 객체 생성 및 실행
if api_key and file_path:
    processor = PDFProcessor(file_path)  # api_key는 내부에서 로드되므로 전달할 필요 없음
    processor.run()
else:
    print("API 키나 파일 경로가 올바르지 않습니다.")
