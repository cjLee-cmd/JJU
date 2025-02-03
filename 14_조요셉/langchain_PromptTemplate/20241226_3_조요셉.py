# 필요한 라이브러리 임포트
import os
from langchain_teddynote.messages import stream_response
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# API 키 설정
api_key = 'your api key'
os.environ['OPENAI_API_KEY'] = api_key

# Prompt 템플릿 정의 (출력 형식 적용)
template = '{country}의 여행하기 좋은 여행지 추천해줘'

# 모델 설정
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    max_tokens=2048,
    temperature=0.1
)

# Chain 생성
prompt_template = PromptTemplate.from_template(template)
chain = prompt_template | model

country = ['대한민국', '일본', '중국']

# 출력 형식 적용
for c in country:
    input_prompt = {'country': c}
    content = chain.invoke(input_prompt).content
    # 출력 형식 적용
    formatted_content = f"추천 여행지 for {c}:\n{content}\n{'='*50}\n"
    print(formatted_content)
