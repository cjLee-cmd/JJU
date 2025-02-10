from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup, SoupStrainer

def load_with_webbase(url):
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs=dict(
            parse_only=SoupStrainer(
                "div",
                attrs={"class": ["newsct_article _article_body", "media_end_head_title"]}
            )
        )
    )
    docs = loader.load()
    return docs[0].page_content[:1000]
