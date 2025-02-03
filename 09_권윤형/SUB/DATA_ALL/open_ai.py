# openai_api.py
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def load_openai_api_key():
    load_dotenv()  # .env 파일 로드
    return os.getenv('OPENAI_API_KEY')

def generate_response(prompt_text, api_key):
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
                       api_key=api_key)

    chain = prompt | model
    return chain.invoke({"result_load": prompt_text}).content
