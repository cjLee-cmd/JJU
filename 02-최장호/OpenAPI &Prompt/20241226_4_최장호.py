from langchain_teddynote.messages import stream_response
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

os.environ['OPENAI_API_KEY'] = '****'


# PromptTemplate 정의
prompt_template = PromptTemplate(
    template="{country1} 수도와 {country2} 수도가 어디인가요?",
    input_variables=['country1', 'country2']
)

# 모델 설정
model = ChatOpenAI(
    model='gpt-3.5-turbo',
    max_tokens=2048,
    temperature=0.1
)

# 체인 구성
chain = prompt_template | model

# 체인 실행
response = chain.invoke({"country1": "한국", "country2": "일본"})

# 결과 출력
print(response)
