import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()

# API 키 설정
api_key = os.environ['OPENAI_API_KEY']

FILE_PATH = 'PDFLoader/data/(제2024-585호)_초광역권_선도기업_선정_및_지원계획_수정공고.pdf'

def show_metadata(docs):
    if docs:
        print("[metadata]")
        print(list(docs[0].metadata.keys()))
        print("\n[examples]")
        max_key_length = max(len(k) for k in docs[0].metadata.keys())
        for k, v in docs[0].metadata.items():
            print(f"{k:<{max_key_length}} : {v}")

def pypdfloader(FILE_PATH):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(FILE_PATH)
    docs = loader.load()
    return docs[0].page_content[:1000], show_metadata(docs)

def pymupdfloader(FILE_PATH):
    from langchain_community.document_loaders import PyMuPDFLoader
    loader = PyMuPDFLoader(FILE_PATH)
    docs = loader.load()
    return docs[0].page_content[:1000], show_metadata(docs)

def pypdfium2(FILE_PATH):
    from langchain_community.document_loaders import PyPDFium2Loader
    loader = PyPDFium2Loader(FILE_PATH)
    docs = loader.load()
    return docs[0].page_content[:1000], show_metadata(docs)

def pdfMiner(FILE_PATH):
    from langchain_community.document_loaders import PDFMinerLoader
    loader = PDFMinerLoader(FILE_PATH)
    docs = loader.load()
    return docs[0].page_content[:1000], show_metadata(docs)

def PdfPlumberLoader(FILE_PATH):
    from langchain_community.document_loaders import PDFPlumberLoader
    loader = PDFPlumberLoader(FILE_PATH)
    docs = loader.load()
    return docs[0].page_content[:1000], show_metadata(docs)

def save_output_to_file(output, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(output)

def evaluate_pdf_split(chain, promp, pdf_loader, model_name, file_path):
    text, matadata = pdf_loader(file_path)
    input_prompt = {'model': model_name, 'matadata' : matadata, 'text': text}
    content = chain.invoke(input_prompt).content
    return f"## Model: {model_name}\n\n### Response:\n{content}\n\n", text

def main(FILE_PATH):
    output = '''
       {score : 90, reason : '문법에 맞게 잘 나눴다. 텍스트의 흐름이 자연스럽고, 문단 구분이 명확하다. 중요한 정보가 잘 보존되었다.'}
       {score : 20, reason : '문법에 맞지 않게 나눴다. 문단이 중간에 끊기거나, 문장이 불완전하게 나뉘었다. 중요한 정보가 누락되었다.'}
       {score : 50, reason : '부분적으로 문법에 맞게 나눴다. 일부 문단은 잘 나뉘었으나, 다른 부분은 문장이 불완전하게 나뉘었다.'}
       {score : 70, reason : '대체로 문법에 맞게 나눴다. 대부분의 문단이 자연스럽게 나뉘었으나, 일부 문장은 불완전하게 나뉘었다.'}
    '''

    promp = PromptTemplate(
        template='''
        {model}을 사용해 pdf를 나눴다.
        
        메타데이터는 아래와 같다.
        {matadata}

        택스트 보고 해당 pdf에 대해 잘 나눴는지 평가한 다음 점수로 환산 하시오. 
        그리고 점수를 준 이유를 설명하시오.
        text : {text}

        결과는 아래와 같아야 합니다.
        score : 점수, reason : '왜 이런 점수를 줬는지에 대한 이유와 감점 요인에 대한 이유'

        ex)
        {output}
        ''',
        input_variables=['model', 'matadata', 'text'],
        partial_variables={'output': output}
    )

    model = ChatOpenAI(
        model="gpt-4o-mini",
        max_tokens=2048,
        temperature=0.1,
        api_key=api_key
    )

    chain = promp | model

    pdf_loaders = [pypdfloader, pymupdfloader, pypdfium2, pdfMiner, PdfPlumberLoader]
    pdf_loader_names = ['pypdfloader', 'pymupdfloader', 'pypdfium2', 'pdfMiner', 'PdfPlumberLoader']
    results = []
    texts = []

    for pdf_loader, name in zip(pdf_loaders, pdf_loader_names):
        result, text = evaluate_pdf_split(chain, promp, pdf_loader, name, FILE_PATH)
        results.append(result)
        texts.append(name+' : \n'+text)

    # 결과물을 Markdown 파일로 저장
    save_output_to_file('\n'.join(results), 'PDFLoader/SCORE.md')
    save_output_to_file('\n'.join(texts), 'PDFLoader/RESULT.txt')

if __name__ == "__main__":
    main(FILE_PATH)