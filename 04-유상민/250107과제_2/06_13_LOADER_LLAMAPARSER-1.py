UPSTAGE_API_KEY = ''

import os


os.environ["OPENAI_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "test_JJU_YuSangMin"

from langchain_upstage import UpstageLayoutAnalysisLoader

file_path = '/content/data/SPRI_AI_Brief_2023년12월호_F.pdf'

loader = UpstageLayoutAnalysisLoader(
    file_path,
    output_type = 'html',
    split = 'page',
    use_ocr = True,
    exclude = ["header", 'footer'],
    api_key= UPSTAGE_API_KEY

)


docs = loader.load()

for doc in docs[6]:
  print(doc)

# -----------------------------------------------

import os
import nest_asyncio

LLAMA_CLOUD_API_KEY = ''
nest_asyncio.apply()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# 파서 설정
parser = LlamaParse(
    result_type="markdown",  # "markdown"과 "text" 사용 가능
    num_workers=8,  # worker 수 (기본값: 4)
    verbose=True,
    language="ko",
    api_key = LLAMA_CLOUD_API_KEY,
)

# SimpleDirectoryReader를 사용하여 파일 파싱
file_extractor = {".pdf": parser}

# LlamaParse로 파일 파싱
documents = SimpleDirectoryReader(
    input_files=["/content/data/SPRI_AI_Brief_2023년12월호_F.pdf"],
    file_extractor=file_extractor,
).load_data()

print(documents[6])