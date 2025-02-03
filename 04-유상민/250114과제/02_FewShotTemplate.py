import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate


os.environ["OPENAI_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "01-01"

example_prompt = PromptTemplate.from_template(
    "Question: \n{question}\nAnswer: \n{answer}"
)
example_prompt

prompt = FewShotPromptTemplate(
    examples = '/workspaces/JJU-1/04-유상민/fewshottemplate/prompt.yaml',
    example_prompt = example_prompt,
    suffix = "Question: \n{input}\nAnswer:",
    input_variables = ["question"],
)
prompt

question = "google이 창립된 연도에 Bill Gates의 나이는 몇살인가요?"

final_prompt = prompt.format(input = question)
final_prompt



llm = ChatOpenAI(
    temperature = 0,
    model_name = "gpt-4o",
)

response = llm.invoke(final_prompt)

chain=prompt | llm
chain_response = chain.invoke({"question": "현재 대한민국 대통령의 운명은?"})
chain_response