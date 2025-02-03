from langchain_openai import ChatOpenAI
from langchain.prompts import load_prompt
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# LLM 설정
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0,
    max_tokens=2000,
    timeout=30,
    max_retries=2,
    api_key=OPENAI_API_KEY
)

# YAML 파일 로드
prompt_yaml = load_prompt(r"C:\Users\eys63\Desktop\기타활동\2024\겨울방학\JJU\01-이연승\과제\25_01_09\PROMPTTEMPLATE.yaml", encoding='utf-8')

# 체인 생성 및 실행
input_data = {"topic": "미래 직업 전망"}
chain = prompt_yaml | llm

# 실행 수정: invoke 메서드 사용
result = chain.invoke(input_data)

# 결과를 파일로 저장
with open("ANSWERFROMLLM.txt", "w", encoding="utf-8") as file:
    file.write(result.content)

print("결과가 ANSWERFROMLLM.txt 파일에 저장되었습니다.")
