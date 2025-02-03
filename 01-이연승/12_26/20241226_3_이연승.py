import os
from langchain_teddynote.messages import stream_response
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Step 1: API 키 설정
# OpenAI API를 사용하기 위해 필요한 인증 정보를 환경 변수로 설정합니다.
os.environ['OPENAI_API_KEY'] = ''

# Step 2: 프롬프트 템플릿 정의
# 질문 형식을 지정하는 템플릿을 만듭니다. {country}는 나중에 입력값으로 대체됩니다.
template = "{country}의 수도는 어디인가요?"
prompt_template = PromptTemplate.from_template(template)

# Step 3: 프롬프트에 값 넣기
# '대한민국'이라는 값을 넣어 질문 문장을 만듭니다.
country = "대한민국"
formatted_prompt = prompt_template.format(country=country)
print("Formatted Prompt:", formatted_prompt)  # 출력: "대한민국의 수도는 어디인가요?"

# Step 4: 모델 초기화
# GPT-3.5-turbo 모델을 설정합니다. max_tokens는 응답 길이, temperature는 응답의 창의성을 조절합니다.
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    max_tokens=2048,
    temperature=0.1
)

# Step 5: 체인 생성
# 프롬프트 템플릿과 모델을 연결합니다. 입력값을 사용해 결과를 생성할 준비를 합니다.
chain = prompt_template | model

# Step 6: 예제 입력과 체인 실행
# 나라 이름을 입력으로 제공해 체인을 실행하고 응답을 받습니다.
input_data = {"country": "대한민국"}
response = chain.invoke(input_data)  # 모델이 응답을 생성합니다.
print("Response:", response)  # 출력: 대한민국의 수도에 대한 설명
