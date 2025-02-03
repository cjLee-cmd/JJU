from langchain_community.document_loaders import PyPDFLoader

FILE_PATH = "과제/data/제5차 국토종합계획(2020~2040)(홈페이지 공개).pdf"

def show_metadata(docs):
    if docs:
        print("[metadata]")
        print(list(docs[0].metadata.keys()))
        print("\n[examples]")
        max_key_length = max(len(k) for k in docs[0].metadata.keys())
        for k, v in docs[0].metadata.items():
            print(f"{k:<{max_key_length}} : {v}")


# 파일 경로 설정
loader = PyPDFLoader(FILE_PATH)

# PDF 로더 초기화
docs = loader.load()

# 문서의 내용 출력
print(docs[10].page_content[:300])

# 메타데이터 출력
show_metadata(docs)