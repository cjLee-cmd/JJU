import os

# 환경 변수 설정
os.environ["OPENAI_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "test_JJU_YuSangMin"

from langchain_openai import ChatOpenAI


# LLM 설정
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

from langchain_core.prompts import load_prompt

# YAML 파일 경로와 인코딩 설정
prompt_yaml = load_prompt("/workspaces/JJU-1/04-유상민/250109과제/PROMPTTABLE.yaml", encoding='utf-8')

# 입력 데이터 정의 및 체인 실행
prompt_yaml.format(technique="인공지능")
chain = prompt_yaml | llm
response = chain.invoke({"technique": "인공지능"})

print(response.content)
