import os
# import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_upstage import UpstageLayoutAnalysisLoader

# 1. 환경 변수 로드
load_dotenv()

# 2. 환경 변수에서 API 키 및 설정 불러오기
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# 3. 필수 키 확인
required_keys = {
    "LLAMA_CLOUD_API_KEY": LLAMA_CLOUD_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "UPSTAGE_API_KEY": UPSTAGE_API_KEY,
    "LANGCHAIN_API_KEY": LANGCHAIN_API_KEY,
    "LANGCHAIN_ENDPOINT": LANGCHAIN_ENDPOINT,
    "LANGCHAIN_PROJECT": LANGCHAIN_PROJECT,
}
missing_keys = [key for key, value in required_keys.items() if not value]
if missing_keys:
    raise ValueError(f".env 파일에 다음 키들이 누락되었습니다: {', '.join(missing_keys)}")

# 4. LangChain 관련 환경 변수 설정
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "")
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# 5. nest_asyncio 적용
# nest_asyncio.apply()

# 6. 비교할 PDF 파일 경로 설정 (사용 환경에 맞춰 수정)
file_path = "C:/Users/koll2/OneDrive/문서/GitHub/JJU3/06-배진우/2024_겨울특강/과제/data/TransUNet.pdf"
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# ------------------------------------------------------------------
# (A) 첫 번째 방식: LlamaParse + SimpleDirectoryReader
# ------------------------------------------------------------------
parser = LlamaParse(
    result_type="markdown",  # 필요 시 "text"로 변경
    num_workers=8,
    verbose=True,
    language="ko",
    api_key=LLAMA_CLOUD_API_KEY,
)

file_extractor = {".pdf": parser}

documents_llama = SimpleDirectoryReader(
    input_files=[file_path],
    file_extractor=file_extractor,
).load_data()

docs_llama = [doc.to_langchain_format() for doc in documents_llama]

# ------------------------------------------------------------------
# (B1) UpstageLayoutAnalysisLoader: split="page"
# ------------------------------------------------------------------
loader_page = UpstageLayoutAnalysisLoader(
    file_path=file_path,
    output_type="text",
    split="page",           # 페이지 단위로 분할
    use_ocr=True,
    exclude=["header", "footer"],  # 헤더/푸터 제외
)
docs_upstage_page = loader_page.load()

# ------------------------------------------------------------------
# (B2) UpstageLayoutAnalysisLoader: split="element"
# ------------------------------------------------------------------
loader_element = UpstageLayoutAnalysisLoader(
    file_path=file_path,
    output_type="text",
    split="element",        # 문서 내 요소 단위로 분할
    use_ocr=True,
    exclude=["header", "footer"],
)
docs_upstage_element = loader_element.load()

# ------------------------------------------------------------------
# 결과를 비교하여 저장
# ------------------------------------------------------------------
current_pyfile = os.path.basename(__file__)  
filename_only = os.path.splitext(current_pyfile)[0]  
# 경로는 사용자 환경에 맞춰 수정해 주세요.
save_path = os.path.join(
    "06-배진우",
    "2024_겨울특강",
    "과제",
    "code",
    "result",
    f"{filename_only}_result.txt"
)

# 결과 파일 쓰기
with open(save_path, "w", encoding="utf-8") as f:
    # (A) LlamaParse 결과
    f.write("=== (A) LlamaParse + SimpleDirectoryReader 결과 ===\n\n")
    for i, doc in enumerate(docs_llama[:3], start=1):
        f.write(f"[문서 {i}]\n{str(doc)}\n\n")

    # (B1) UpstageLayoutAnalysisLoader (page 단위) 결과
    f.write("=== (B1) UpstageLayoutAnalysisLoader (split='page') 결과 ===\n\n")
    for i, doc in enumerate(docs_upstage_page[:3], start=1):
        f.write(f"[문서 {i}]\n{str(doc)}\n\n")

    # (B2) UpstageLayoutAnalysisLoader (element 단위) 결과
    f.write("=== (B2) UpstageLayoutAnalysisLoader (split='element') 결과 ===\n\n")
    for i, doc in enumerate(docs_upstage_element[:3], start=1):
        f.write(f"[문서 {i}]\n{str(doc)}\n\n")

print(f"[INFO] 결과가 다음 파일에 저장되었습니다: {save_path}")
