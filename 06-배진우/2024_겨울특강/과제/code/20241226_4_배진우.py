from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

# os.environ['OPENAI_API_KEY'] = ""


template = "{country1}과 {country2}의 수도는 각각 어디인가요? {country3}의 수도는 어디인가요?"

prompt = PromptTemplate(
    template=template,
    input_variables=["country1"],
    partial_variables={
        "country2": "미국",
        "country3": "일본"
    },
)

model = ChatOpenAI(
    model="gpt-4o",
    max_tokens=2048,
    temperature=0.1
)

chain = prompt | model


result = chain.invoke({
    "country1": "대한민국",
    "country2": "호주",
    "country3": "독일"
}).content

print(result)