import os
from dotenv import load_dotenv
import yaml
from datetime import datetime

# .env 파일 로드
load_dotenv()

# 환경 변수 불러오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

from langchain_openai import ChatOpenAI
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.messages import stream_response

def load_prompt_from_yaml(yaml_file_path: str):
    with open(yaml_file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    examples_data = data.get('examples', [])
    suffix = data.get('suffix', '')

    # example_prompt 구성
    example_prompt = PromptTemplate.from_template(
        "Question:\n{question}\nAnswer:\n{answer}"
    )

    # FewShotPromptTemplate 생성
    prompt = FewShotPromptTemplate(
        examples=examples_data,
        example_prompt=example_prompt,
        suffix=suffix,
        input_variables=["question"],
    )
    return prompt

def main():
    # YAML 파일로부터 prompt 로드
    prompt = load_prompt_from_yaml("06-배진우/2024_겨울특강/과제/data/prompt.yaml")

    # ChatOpenAI 객체 생성
    llm = ChatOpenAI(
        temperature=0, 
        model_name="gpt-4o",  # 사용하고자 하는 모델명
        openai_api_key=OPENAI_API_KEY  # .env에서 가져온 key
    )

    # 예시 question
    question = "Google이 창립된 연도에 Bill Gates의 나이는 몇 살인가요?"
    formatted_prompt = prompt.format(question=question)

    # 체인을 만들고 실행 (간단한 방식)
    chain = prompt | llm

    # 결과 추출
    result = chain.invoke({"question": question})
    print(result.content)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("06-배진우/2024_겨울특강/과제/code/result/readme.md", "a", encoding="utf-8") as f:
        f.write("\n\n## Run Result at {}\n".format(timestamp))
        f.write(f"**Question**: {question}\n\n**Answer**:\n{result.content}\n")

if __name__ == "__main__":
    main()
