#=========page=========
import os

os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'
os.environ['LANGCHAIN_API_KEY'] = 'LANGCHAIN_API_KEY' # 본인 api
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'test_JJU_KwonYunHyeong'  # 본인 프로젝트

from langchain_upstage import UpstageLayoutAnalysisLoader

file_path = '/content/data/ko_pdf.pdf'

loader = UpstageLayoutAnalysisLoader(
    file_path=file_path,
    output_type='html', # html, text
    split = 'page', #page, element
    use_ocr=True,
    exclude = ['header', 'footer'],
    api_key=UPSTAGE_API_KEY
)

docs = loader.load()

#print(len(docs))
print(docs[0].page_content)
#print(docs[0].metadata)

#=========element=========
from langchain_upstage import UpstageLayoutAnalysisLoader

file_path = '/content/data/ko_pdf.pdf'

loader = UpstageLayoutAnalysisLoader(
    file_path=file_path,
    output_type='html', # html, text
    split = 'element', #page, element
    use_ocr=True,
    exclude = ['header', 'footer'],
    api_key=UPSTAGE_API_KEY
)

docs = loader.load()

#print(len(docs))
print(docs[0].page_content)
#print(docs[0].metadata)