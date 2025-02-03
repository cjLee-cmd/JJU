import os
import nest_asyncio
from llama_parse import LlamaParse
from langchain_upstage import UpstageLayoutAnalysisLoader

# 파일 경로 설정
file_path = "파일 경로 넣기"

# API Keys 설정
LLAMA_CLOUD_API_KEY = 'LLAMA API 키 넣기'
os.environ["OPENAI_API_KEY"] = 'OpenAI API 키 넣기'
os.environ["UPSTAGE_API_KEY"] = 'UPSTAGE API 키 넣기'
nest_asyncio.apply()

def test_llama_parser():
    print("\n=== LlamaParse 결과 ===")
    llama_parser = LlamaParse(
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name="openai-gpt4o",
        vendor_multimodal_api_key=os.environ["OPENAI_API_KEY"],
        result_type="markdown",
        language="ko",
        api_key=LLAMA_CLOUD_API_KEY,
        page_separator="/n=================/n",
    )
    
    parsed_docs = llama_parser.load_data(file_path=file_path)
    docs = [doc.to_langchain_format() for doc in parsed_docs]
    print(f"문서 수: {len(docs)}")
    print("첫 번째 문서 내용 샘플:")
    print(docs[0].page_content[:500])  # 처음 500자만 출력

def test_upstage_parser():
    print("\n=== Upstage Parser 결과 ===")
    # 페이지 단위 파싱
    loader = UpstageLayoutAnalysisLoader(
        file_path,
        output_type="text",
        split="page",
        use_ocr=True,
        exclude=["header", "footer"],
    )
    
    docs = loader.load()
    print(f"문서 수 (페이지 단위): {len(docs)}")
    print("첫 번째 문서 내용 샘플:")
    print(docs[0].page_content[:500])  # 처음 500자만 출력

if __name__ == "__main__":
    test_llama_parser()
    test_upstage_parser()