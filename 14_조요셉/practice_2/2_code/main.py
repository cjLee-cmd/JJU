# 필요한 라이브러리 임포트
import os
from datetime import datetime
from langchain_teddynote.messages import stream_response
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

from sorter import load_and_split

load_dotenv()

output = '''
    {input file type : pdf, 
    Loader : py_pdf_loader,
    splitter : charcter_text_splitter,
    socre : 73,
    result : loader로 llm이 이해할 수 있게 불러왔으며, spiltter가 llm이 이해할 수 있게 잘 나눴다.
    }
    {
    "input file type": "csv",  
    "Loader": "pandas_loader",  
    "splitter": "row_splitter",  
    "score": 90,  
    "result": "loader가 데이터를 DataFrame으로 잘 로드했고, splitter가 행 단위로 적절히 나눴다."
    }
    {
    "input file type": "html",  
    "Loader": "beautifulsoup_loader",  
    "splitter": "tag_based_splitter",  
    "score": 88,  
    "result": "HTML 문서를 문제없이 파싱했으며, splitter가 태그 단위로 적절히 나눴다."
    }
    {
    "input file type": "pdf",  
    "Loader": "pdf_loader",  
    "splitter": "word_splitter",  
    "score": 32,  
    "result": "loader가 PDF 파일을 제대로 읽지 못했고, splitter가 단어 단위로 잘못된 위치에서 나눴다."
    }

'''

travle_promp = PromptTemplate(
    template='''
    당신은 지금부터 llm 평가원 입니다.
    주어지는 결과를 보고 이에 대해서 평가 한 이후, 왜 그렇게 평가 했는지에 대해서 설명하시오.
    결과는 어떠한 파일을 langchain의 loader를 사용해 불러왔으며, 이를 langchain의 splitter를 사용해 나눈 것이다.
    결과는 여러 방법으로 load하고, 여러 방법으로 split한 것이다. 주어진 각각의 모든 방법을 평가 해라.
    이때 llm이 splitter한 파일을 잘 이해할 수 있는지에 대해서 평가를 하면 된다.
    평가 방법아래와 같다.
    input file type : 어떤 확장자 파일이 들어왔는지
    Loader : langchain의 어떤 라이브러리를 사용해서 load했는지
    splitter : langchain의 어떤 라이브러리를 사용해서 spilt했는지
    score : 평가 점수
    result : 이유(적절하고 직관적으로 설명)

    ex) {output}

    {results}
    ''',
    input_variables=['results'],
    partial_variables={'output' : output}
)

model = ChatOpenAI(
    model="gpt-4o",
    max_tokens=2048,
    temperature=0.1
)

chain = travle_promp | model

file_path = "14_조요셉/practice_2/1_usedData/240327_공고_아이비케이벤처투자의 여신전문금융업 등록.pdf"
results = load_and_split(file_path)

input_prompt = {'results':results}
content = chain.invoke(input_prompt).content

prompt = travle_promp.format(results=results)
print(prompt+'\n\n')

print(content)