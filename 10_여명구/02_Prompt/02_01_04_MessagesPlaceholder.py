import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# 1. 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# 2. ChatPromptTemplate 정의
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 요약 전문 AI 어시스턴트입니다. 당신의 임무는 주요 키워드로 대화를 요약하는 것입니다."),
    MessagesPlaceholder(variable_name="conversation"),
    ("human", "지금까지의 대화를 {word_count} 단어로 요약합니다."),
])

# 3. LLM 모델 초기화
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=OPENAI_API_KEY,
)

# 4. 체인 생성
chain = chat_prompt | llm

# 5. 입력 데이터 준비
input_data = {
    "word_count": 5,
    "conversation": [
        ("human", "안녕하세요! 저는 오늘 새로 입사한 테디 입니다. 만나서 반갑습니다."),
        ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
    ],
}

# 6. 체인 실행 및 결과 출력
response = chain.invoke(input_data)
print(response.content)

print("="*80)

# content를 제외한 나머지 출력
print({key: value for key, value in response.__dict__.items() if key != 'content'})