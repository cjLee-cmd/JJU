from langchain_teddynote.messages import stream_response
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

os.environ['OPENAI_API_KEY'] = '****'

# PromptTemplate 정의
prompt_template_1 = PromptTemplate.from_template("{country}의 수도는 어디인가요?")

# 모델 설정
model = ChatOpenAI(
    model='gpt-3.5-turbo',
    max_tokens=2048,
    temperature=0.1
)

# 체인 구성
chain = prompt_template_1 | model

# 체인 실행
response = chain.invoke({"country": "대한민국"})

# 결과 출력
print(response)
