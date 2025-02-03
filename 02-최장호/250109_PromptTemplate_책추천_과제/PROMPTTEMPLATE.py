import os
from langchain_openai import ChatOpenAI
from langchain.prompts import load_prompt  
from langchain.chains import LLMChain  

os.environ['OPENAI_API_KEY'] = '***'
os.environ['LANGCHAIN_API_KEY'] = '***'  
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'test_JJU_Agent_02'

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  

llm = ChatOpenAI(
    model="gpt-4",  
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY
)


prompt_yaml_1 = load_prompt("C:/Users/kowm6/Desktop/PROMPTTEMPLATE.yaml", encoding="utf-8")

chain = LLMChain(prompt=prompt_yaml_1, llm=llm)

result = chain.run({
    "age_group": "청소년",
    "genre": "판타지"
})


with open("answerfromllm.txt", "w", encoding="utf-8") as file:
    file.write(result)

print("결과가 answerfromllm.txt 파일에 저장되었습니다.")
