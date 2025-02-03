import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=('https://n.news.naver.com/article/023/0003880611?cds=news_media_pc&type=editn',),
    bs_kwargs = dict(
        parse_only = bs4.SoupStrainer(
            'div',
            attrs = {'class' : ['newsct_article _article_body', 'media_end_head_title']},
            #attrs = {'class' : ['media_end_head_title'], 'id' : ['newsct_article']},
        )
    ),

    header_template = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
     }
)

docs = loader.load()
print(f'문서의 수 : {len(docs)}')
print(docs)