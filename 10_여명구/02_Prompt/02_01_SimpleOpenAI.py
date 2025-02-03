import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ======================
# 환경 변수 설정
# ======================
os.environ["OPENAI_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "test_JJU_YEOMYEONGGU"

# ======================
# LangChain OpenAI 모델 초기화
# ======================
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# ======================
# Prompt Template 정의
# ======================
template = "{input_1}과 {input_2}, {input_3}에 대해 각각 한문장씩 간략하게 설명해줘."
prompt_2 = PromptTemplate(
    template=template,
    input_variables=["input_1", "input_2"],
    partial_variables={"input_3": "인공지능"},  # input_3 기본값 설정
)

# ======================
# Prompt Format 및 Partial 사용 예시
# ======================
prompt_partial = prompt_2.partial(input_3="짐")
chain_2 = prompt_partial | llm
result_1 = chain_2.invoke({"input_1": "러닝머신신", "input_2": "머신러닝"})
print("Partial Prompt 결과:")
print(result_1.content)

# ======================
# 날짜 관련 Prompt
# ======================
print("\n오늘 날짜를 출력:")
print(datetime.now().strftime("%B %d"))

prompt = PromptTemplate(
    template="오늘의 날짜는 {today} 입니다. 오늘이 생일인 유명인 {n}명을 나열해 주세요. 생년월일을 표기해주세요.",
    input_variables=["n"],
    partial_variables={"today": datetime.now().strftime("%B %d")},
)

chain = prompt | llm
result_2 = chain.invoke(4)
print("\n오늘의 생일자 결과:")
print(result_2.content)
