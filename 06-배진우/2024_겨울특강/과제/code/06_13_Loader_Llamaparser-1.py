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

# 1) LlamaParse 초기화 (result_type="markdown" -> 최신 버전에서 문제가 될 수 있으므로 "text" 권장)
parser = LlamaParse(
    result_type="markdown",  # "markdown", "text" 중 선택 가능. (에러 시 "text"로 변경)
    num_workers=8,
    verbose=True,
    language="ko",
    api_key=LLAMA_CLOUD_API_KEY,
)

file_extractor = {".pdf": parser}

# 2) SimpleDirectoryReader로 문서 로드
documents_llama = SimpleDirectoryReader(
    input_files=[file_path],
    file_extractor=file_extractor,
).load_data()

# 3) LangChain 형식으로 변환
docs_llama = [doc.to_langchain_format() for doc in documents_llama]

# ------------------------------------------------------------------
# (B) 두 번째 방식: UpstageLayoutAnalysisLoader
# ------------------------------------------------------------------

loader = UpstageLayoutAnalysisLoader(
    file_path=file_path,
    output_type="text",   # "text" 또는 "json" 등
    split="page",         # 페이지 단위로 분할
    use_ocr=True,         # OCR 사용
    exclude=["header", "footer"],  # 헤더/푸터 제외
)

docs_upstage = loader.load()

# ------------------------------------------------------------------
# 두 결과를 비교하여 저장
# ------------------------------------------------------------------

# 현재 .py 파일 이름에 _result.txt를 붙여서 저장
current_pyfile = os.path.basename(__file__)  # 예: "test.py"
filename_only = os.path.splitext(current_pyfile)[0]  # 예: "test"
save_dir = "."  # 원하는 디렉터리를 지정할 수 있습니다. (상대/절대 경로 가능)
save_path = os.path.join(save_dir, f"06-배진우/2024_겨울특강/과제/code/result/{filename_only}_result.txt")

# 파일에 쓰기
with open(save_path, "w", encoding="utf-8") as f:
    # (A) LlamaParse 결과
    f.write("=== (A) LlamaParse + SimpleDirectoryReader 결과 ===\n\n")
    for i, doc in enumerate(docs_llama[:3], start=1):
        f.write(f"[문서 {i}]\n{str(doc)}\n\n")

    # (B) UpstageLayoutAnalysisLoader 결과
    f.write("=== (B) UpstageLayoutAnalysisLoader 결과 ===\n\n")
    for i, doc in enumerate(docs_upstage[:3], start=1):
        f.write(f"[문서 {i}]\n{str(doc)}\n\n")

print(f"[INFO] 결과가 다음 파일에 저장되었습니다: {save_path}")
