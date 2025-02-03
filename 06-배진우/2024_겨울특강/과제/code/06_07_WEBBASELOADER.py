import os
import bs4
import urllib3
from langchain_community.document_loaders import WebBaseLoader

# USER_AGENT 환경 변수 설정
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " \
                          "AppleWebKit/537.36 (KHTML, like Gecko) " \
                          "Chrome/102.0.0.0 Safari/537.36"

# SSL 인증서 검증 비활성화가 필요한 경우 경고 숨기기 (권장하지 않음)
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

loader = WebBaseLoader(
    web_paths=[
        "https://www.gmarket.co.kr/n/search?keyword=rtx4090",
        "https://www.gmarket.co.kr/n/search?keyword=ryzen+9800x3d",
        "https://www.gmarket.co.kr/n/search?keyword=x870e"
    ],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
    header_template={
        "User-Agent": os.environ["USER_AGENT"],
    },
)

# SSL 인증서 검증 비활성화가 필요한 경우 (권장하지 않음)
loader.requests_kwargs = {"verify": False}

# 데이터 로드
docs = loader.load()

# 디버깅: 로드된 문서 수 및 각 문서의 콘텐츠 길이 확인
print(f"로드된 문서 수: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"문서 {i}의 콘텐츠 길이: {len(doc.page_content)}")
    print(doc.page_content[:500])
    print("===" * 10)
