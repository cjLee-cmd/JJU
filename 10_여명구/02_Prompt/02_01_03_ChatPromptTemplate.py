import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. LLM 모델 초기화
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)

# 2. 프롬프트 템플릿 정의
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 내용을 요약하는 어시스턴트야. 너의 이름은 {name}이야."),
    ("human", "반가워"),
    ("ai", "안녕하세요. 무엇을 도와드릴까요?"),
    ("human", "{user_input}"),
])

# 3. 메시지 포맷팅 및 체인 생성
# messages = chat_prompt.format_messages(
#     name="앤디",
#     user_input="옵시디언이라는 메모 어플리케이션 설명해줘."
# )

# print(llm.invoke(messages).content)

chain = chat_prompt | llm

# 4. 체인 실행
response = chain.invoke({
    "name": "앤디",
    "user_input": "옵시디언이라는 메모 어플리케이션 설명해줘."
})

# 5. 결과 출력
# content만 출력
print(response.content)

print("="*80)

# content를 제외한 나머지 출력
print({key: value for key, value in response.__dict__.items() if key != 'content'})