import os

os.environ['OPENAI_API_KEY'] = 'API_KEY'


from langchain_core.prompts import PromptTemplate

#template 정의
template = "{country}의 수도는 어디인가요?"

#from_template 메소드를 이용하여 PromptTemplate 객체 생성
prompt_template = PromptTemplate.from_template(template)

#prompt 생성
prompt = prompt_template.format(country="대한민국")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model_name="gpt-3.5-turbo",
                   max_tokens=2048,
                   temperature=0.1)

#prompt를 PromptTemplate 객체로 생성합니다.
prompt = PromptTemplate.from_template('{country}에 대해 쉽게 설명해주세요')

model = ChatOpenAI(model_name="gpt-3.5-turbo",
                   max_tokens=2048,
                   temperature=0.1)

chain = prompt | model

input = {'country':'대한민국'}

print(chain.invoke(input).content)