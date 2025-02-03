import os

os.environ['OPENAI_API_KEY'] = 'API_KEY'


from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model_name="gpt-3.5-turbo",
                   max_tokens=2048,
                   temperature=0.1)

template = "{country1}과 {country2}의 수도는 각각 어디인가요?"

prompt = PromptTemplate(
    template=template,
    input_variables=["country1"],
    partial_variables={"country2":"미국"} #dictionary 형태
)


chain = prompt | model

print(chain.invoke({"country1":"대한민국"}).content)

prompt_partial = prompt.partial(country2="캐나다")
chain = prompt_partial | model

print(chain.invoke({"country1":"대한민국"}).content)
print(chain.invoke({"country1":"대한민국", "country2":"호주"}).content)

prompt_partial2 = prompt.partial(country1="일본", country2="프랑스")
chain2 = prompt_partial2 | model
print(chain2.invoke({}).content)
print(chain2.invoke({"country1":"대한민국"}).content)

