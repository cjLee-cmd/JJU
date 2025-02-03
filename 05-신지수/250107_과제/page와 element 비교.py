from langchain_upstage import UpstageLayoutAnalysisLoader
import os

# 파일 경로
file_path = "파일 경로 넣기"

# API 키 설정
os.environ["UPSTAGE_API_KEY"] = 'API 키 넣기'

# -------------------------------------------------------------------
# 문서 로더 설정
loader = UpstageLayoutAnalysisLoader(
    file_path,
    output_type="text",
    split="page",
    use_ocr=True,
    exclude=["header", "footer"],
)

# 문서 로드
docs = loader.load()

# 결과 출력
for doc in docs[:3]:
    print(doc)

# -------------------------------------------------------------------
# 문서 로더 설정
loader = UpstageLayoutAnalysisLoader(
    file_path,
    output_type="text",
    split="element",
    use_ocr=True,
    exclude=["header", "footer"],
)

# 문서 로드
docs = loader.load()

# 결과 출력
for doc in docs[:3]:
    print(doc)