import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)


from langchain_core.prompts import load_prompt

prompt = load_prompt("10_여명구\\data\\my_QnA.yaml", encoding="utf-8")
Prompt_1 = prompt.format(input="맥북 에어")
chain = prompt | llm

print(chain.invoke(prompt).content) # input 없이 그냥 yaml 파일만 LLM에 넣은 결과를 출력

print("="*80)

print(chain.invoke(Prompt_1).content) # input : 맥북 에어를 넣은 결과를 출력

print("="*80)

print(chain.invoke("맥북 프로").content) # input : 맥북 프로를 넣은 결과를 출력