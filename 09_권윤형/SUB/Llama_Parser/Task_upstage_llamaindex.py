from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
UPSTAGE_API_KEY = os.getenv('UPSTAGE_API_KEY')
LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

documents = LlamaParse(
    use_vendor_mulitmodal_model = True,
    vendor_multimodal_model_name = 'openai-gpt4o',
    vendor_multimodal_api = OPENAI_API_KEY,
    api_key=LLAMA_CLOUD_API_KEY,
    result_type='markdown',
    language='ko',
)

parsed_docs = documents.load_data(file_path='./09_권윤형/SUB/Llama_Parser/ko_pdf.pdf')

docs = [doc.to_langchain_format() for doc in parsed_docs]

# print(docs[0].page_content)
# print('====='*20)

#=========Upstage=========

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'test_JJU_KwonYunHyeong'  # 본인 프로젝트

from langchain_upstage import UpstageLayoutAnalysisLoader

file_path = './09_권윤형/SUB/Llama_Parser/ko_pdf.pdf'

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