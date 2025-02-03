FILE_PATH = r'C:\Users\kyh97\OneDrive\문서\GitHub\test_JJU\pdf_lecture\2024년도_전략기술_테마별_프로젝트(DCP)_제2차_기술수요조사_시행계획_공고.pdf'

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import PDFPlumberLoader

loader_pypdf = PyPDFLoader(FILE_PATH)
loader_pymupdf = PyMuPDFLoader(FILE_PATH)
loader_pypdfium2 = PyPDFium2Loader(FILE_PATH)
loader_pdfminer = PDFMinerLoader(FILE_PATH)
loader_plumber = PDFPlumberLoader(FILE_PATH)

#pdf loader 초기화
docs_pypdf     = loader_pypdf.load()
docs_pymupdf   = loader_pymupdf.load()
docs_pypdfium2 = loader_pypdfium2.load()
docs_pdfminer  = loader_pdfminer.load()
docs_plumber   = loader_plumber.load()

docs_pypdf_500 = docs_pypdf[0].page_content[:500]
docs_pymupdf_500 = docs_pymupdf[0].page_content[:500]
docs_pypdfium2_500 = docs_pypdfium2[0].page_content[:500]
docs_pdfminer_500 = docs_pdfminer[0].page_content[:500]
docs_plumber_500 = docs_plumber[0].page_content[:500]


output_file = "./pdf_lecture/PDF_Loader_result.txt"
with open(output_file, "w", encoding="utf-8") as out_file:
    out_file.write("=== PyPDFLoader ===\n")
    out_file.write(docs_pypdf_500 + "\n\n")
    
    out_file.write("=== PyMuPDFLoader ===\n")
    out_file.write(docs_pymupdf_500 + "\n\n")
    
    out_file.write("=== PyPDFium2Loader ===\n")
    out_file.write(docs_pypdfium2_500 + "\n\n")
    
    out_file.write("=== PDFMinerLoader ===\n")
    out_file.write(docs_pdfminer_500 + "\n\n")
    
    out_file.write("=== pdfplumber ===\n")
    out_file.write(docs_plumber_500 + "\n\n")

#print(f"결과물이 '{output_file}'에 저장되었습니다.")

# with open("./pdf_lecture/PDF_Loader_result.txt") as f:
#   result_load = f.read()
with open("./pdf_lecture/PDF_Loader_result.txt", "r", encoding="utf-8") as f:
    result_load = f.read()


from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#from langchain_teddynote.messages import stream_response #스트리밍 출력
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

template = '''
pdf 파일을 여러 라이브러리로 로드한 후,
500개만 출력해서 저장한 txt 파일이 {result_load}인데, 
어떤 라이브러리를 사용했을 때, RAG로 사용하기 처리가 어느정도로 잘 되었는지 
여러 요소로 정량 평가한 숫자를 총점을 포함한 표로 만들어 주고, 이유를 설명해줘
'''

prompt = PromptTemplate(
    template=template,
    input_variables=["result_load"]
)

model = ChatOpenAI(model_name="gpt-4o-mini",
                   max_tokens=2048,
                   temperature=0.1,
                   api_key = OPENAI_API_KEY)

chain = prompt | model

# country 변수에 입력된 값이 자동으로 치환되어 수행됨
print(chain.invoke({"result_load":result_load}).content)