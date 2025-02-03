import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# .env 파일 로드
load_dotenv()

# 환경 변수 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

llm= ChatOpenAI(
    model = "gpt-4o",
    max_tokens = None,
    timeout = None,
    max_retries = 1,
    api_key = OPENAI_API_KEY
)

template = "{input}의 개념은?"


prompt_3 = PromptTemplate.from_template(template)

prompt_3 = prompt_3.format(input=input())

result_3 = llm.invoke(prompt_3)

print(result_3.content)
