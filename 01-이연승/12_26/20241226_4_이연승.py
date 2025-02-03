import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Step 1: API Key 설정
os.environ['OPENAI_API_KEY'] = ''

# Step 2: 템플릿 정의
template = "장르가 {genre}이고, {age_group}에 적합한 영화를 추천해 주세요. 추천 영화는 {n}개를 나열하고, 간단한 줄거리도 포함해 주세요."
prompt_template = PromptTemplate.from_template(template)

# Step 3: partial로 두 변수 고정
partial_prompt = prompt_template.partial(genre="액션", age_group="청소년")

# Step 4: OpenAI 모델 설정
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    max_tokens=2048,
    temperature=0.3  # 낮은 온도로 더 정확한 응답 유도
)

# Step 5: 남은 변수(n)를 실행 시 제공
n = 3  # 추천할 영화 개수
final_prompt = partial_prompt.format(n=n)

print("\nFinal Prompt (전송 전):")
print(final_prompt)

# Step 6: OpenAI 모델 호출
response = model.invoke(final_prompt)

# Step 7: 응답 출력
print("\nGenerated Response:")
print(response.content)
