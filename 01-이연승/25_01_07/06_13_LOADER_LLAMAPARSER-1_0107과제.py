#!pip install -U langchain_upstage
#!pip install -qU langchain-teddynote
print("===========upstage============")
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
upstage_api_key = os.getenv("UPSTAGE_API_KEY")

from langchain_upstage import UpstageLayoutAnalysisLoader

# 파일 경로
file_path = r"C:\Users\eys63\Desktop\기타활동\2024\겨울방학\24Winter_Vacation\data\SPRI_AI_Brief_2023년12월호_F.pdf"

# test
from dotenv import load_dotenv
load_dotenv(dotenv_path=r"C:\Users\eys63\Desktop\기타활동\2024\겨울방학\24Winter_Vacation\25_01_07\.env")

# 문서 로더 설정
loader = UpstageLayoutAnalysisLoader(
    file_path,
    output_type="html",
    split="page",
    use_ocr=True,
    exclude=["header", "footer"],
    api_key= upstage_api_key
)

# 문서 로드
docs = loader.load()

# 결과 출력
for doc in docs[:]:
    print(doc)

print("===========LLAMAPARSER============")

# !pip install llama-index-core llama-parse llama-index-readers-file python-dotenv
# !pip install -qU langchain-community
import os
import nest_asyncio

LLAMA_CLOUD_API_KEY = ''
nest_asyncio.apply()

from dotenv import load_dotenv
import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

    
# .env 파일 로드
load_dotenv(),

# 환경 변수에서 API 키 가져오기
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# 파서 설정
parser = LlamaParse(
    result_type="markdown",  # "markdown"과 "text" 사용 가능
    num_workers=8,  # worker 수 (기본값: 4)
    verbose=True,   # 상세 정보 출력 (진행중인 내용 출력)
    language="ko",  # 언어 설정 (기본값: "en")
)

# SimpleDirectoryReader를 사용하여 파일 파싱
file_extractor = {".pdf": parser}

# LlamaParse로 파일 파싱
documents = SimpleDirectoryReader(
    input_files=[r"C:\Users\eys63\Desktop\기타활동\2024\겨울방학\24Winter_Vacation\data\SPRI_AI_Brief_2023년12월호_F.pdf"],
    file_extractor=file_extractor,
).load_data()

# 랭체인 도큐먼트로 변환
docs = [doc.to_langchain_format() for doc in documents]

# markdown 형식으로 추출된 테이블 확인
print(docs[-2].page_content)
