import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 파일 경로
file_path = '/workspaces/JJU-1/04-유상민/WEEK4/04_PYDENTIC/E-MAIL.txt'

# 파일 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    email_conversation = file.read()

# 이메일 요약용 데이터 모델
class EmailSummary(BaseModel):
    person: str = Field(description="The sender of the email")
    email: str = Field(description="The email address of the sender")
    subject: str = Field(description="The subject of the email")
    summary: str = Field(description="A summary of the email content")
    date: str = Field(description="The meeting date and time mentioned in the email content")
    Is_Spam: str = Field(description="Is the email spam or not. If it's spam, enter '스팸'; otherwise, 'no spam'.")

# Output Parser
parser = PydanticOutputParser(pydantic_object=EmailSummary)
parser.get_format_instructions()
# Prompt 템플릿
prompt = PromptTemplate.from_template(
    """
You are a helpful assistant.

QUESTION:
{question}

EMAIL CONVERSATION:
{email_conversation}

FORMAT:
{format}
"""
)
prompt = prompt.partial(format=parser.get_format_instructions())

# LLM 초기화
llm = ChatOpenAI(model='gpt-4o', temperature=0)

# 체인 생성
chain = prompt | llm | parser

response = chain.invoke(
    {
        "email_conversation": email_conversation,
        "question": """
        이메일을 다음과 같이 key - value 형식으로 요약해.
        "person": 
        "email": 
        "subject": 
        "summary": 
        "date": 
        "Is_Spam": 
        """,
        "format": parser.get_format_instructions(),
    }
)

# Pydantic 모델을 딕셔너리 형태로 변환
email_summary_dict = response.dict()  # response를 딕셔너리로 변환

# 결과 출력
print("Extracted Information:")
for key, value in email_summary_dict.items():
    print(f"{key}: {value}")

# 결과를 JSON 형식으로 출력
print(json.dumps(email_summary_dict, ensure_ascii=False, indent=4))

# 결과를 JSON 파일로 저장
output_file = 'reply.json'
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(email_summary_dict, file, ensure_ascii=False, indent=4)

print(f"Reply saved to {output_file}.")