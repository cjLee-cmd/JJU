from langchain_teddynote.messages import stream_response
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from datetime import datetime
import os

os.environ['OPENAI_API_KEY'] = '****'

datetime.now().strftime('%Y%m%d')

def get_today():
    return datetime.now().strftime('%Y%m%d')


# PromptTemplate 정의
prompt_template = PromptTemplate(
    template="{today} 생일인 유명인 {n} 명 알려주세요",
    input_variables=['n'],
    partial_variables={'today': get_today()}
)


# 모델 설정
model = ChatOpenAI(
    model='gpt-4-turbo',
    max_tokens=2048,
    temperature=0.1
)

# 체인 구성
chain = prompt_template | model

# 체인 실행
response = chain.invoke({"n" : "3"})

# 결과 출력
print(response)
