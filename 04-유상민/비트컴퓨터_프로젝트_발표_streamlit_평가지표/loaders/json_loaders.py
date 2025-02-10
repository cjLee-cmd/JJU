from langchain_community.document_loaders import JSONLoader as JSONLoaderClass

def load_with_json(file_path):
    loader = JSONLoaderClass(
        file_path=file_path,
        jq_schema='.',  # 전체 JSON 로드
        text_content=False
    )
    docs = loader.load()
    return docs[0].page_content[:1000]
