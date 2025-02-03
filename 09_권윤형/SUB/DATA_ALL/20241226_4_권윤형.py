import os

os.environ['OPENAI_API_KEY'] = 'API_KEY'


from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model_name="gpt-4o",
                   max_tokens=2048,
                   temperature=0.1)

template = "{country1}과 {country2}과 {country3}의 수도는 각각 어디인가요?"

prompt = PromptTemplate(
    template=template,
    input_variables=["country1"],
    partial_variables={"country2":"미국", "country3":"프랑스"} #dictionary 형태
)

prompt_partial = prompt.partial(country2="캐나다", country3="중국")

chain = prompt_partial | model

print(chain.invoke({"country1":"대한민국"}).content)


