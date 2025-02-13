# config.py
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# API 키 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 파일 확장자와 MIME 타입 매핑 - PDF 파일만 사용
EXTENSION_MIME_MAPPING = {'.pdf': 'application/pdf'}

# 모델 설정
MODEL_NAME = "gpt-4o-mini"