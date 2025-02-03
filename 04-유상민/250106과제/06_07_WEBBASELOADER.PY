import bs4
from langchain_community.document_loaders import WebBaseLoader

# WebBaseLoader 초기화
loader = WebBaseLoader(
    web_paths=['https://smartstore.naver.com/toptime/products/6345008331'],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            ["h3", "div"],
            attrs={"class": ["product_title__1XYZ", "product_detail__2ABC"]},
        )
    ),
    header_template={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
)

# 문서 로드
docs = loader.load()

# 문서의 수 확인 및 출력
print(f"문서의 수: {len(docs)}")
print(docs)
