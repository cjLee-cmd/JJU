import os
from langchain_teddynote.messages import stream_response
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_KEY'] = "my_key"

# 프롬프트 템플릿 정의 (국가를 위한 플레이스홀더 포함)
template = "{country}의 수도는 어디인가요?"

# PromptTemplate 인스턴스 생성 시 키워드 인자 사용
prompt_template = PromptTemplate(
    template=template,
    input_variables=["country"]
)

model = ChatOpenAI(
    model = "gpt-4o",
    max_tokens = 2048,
    temperature = 0.1
)

prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 설명해주세요.")
model = ChatOpenAI()
chain = prompt | model
input = {"topic": "인공지능 모델의 학습 원리"}
print(chain.invoke(input))
