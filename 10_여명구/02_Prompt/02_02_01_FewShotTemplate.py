# 필요한 라이브러리 임포트
import os
import yaml  # YAML 파일 처리를 위한 라이브러리
from dotenv import load_dotenv  # 환경 변수 로드용
from langchain_openai import ChatOpenAI  # OpenAI와 연결
from langchain_core.prompts.few_shot import FewShotPromptTemplate  # FewShot 템플릿
from langchain_core.prompts import PromptTemplate  # 일반 템플릿

# 1. 환경 변수 로드
load_dotenv()  # .env 파일 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API 키 가져오기
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")  # LangChain API 키
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")  # LangChain 추적 설정
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")  # LangChain 엔드포인트
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")  # LangChain 프로젝트 이름

# 2. ChatOpenAI 객체 생성
llm = ChatOpenAI(
    temperature=0,  # 창의성 제어 (0: 가장 사실적인 응답)
    model_name="gpt-4o",  # 사용할 모델 이름
    api_key=OPENAI_API_KEY,
)

# 3. Examples 데이터 로드 (YAML 파일 읽기)
with open(r"10_여명구\02_Prompt\data\examples.yaml", "r", encoding="utf-8") as f:
    yaml_data = yaml.safe_load(f)  # YAML 파일 로드
    examples = yaml_data["examples"]  # "examples" 키에 있는 데이터 가져오기

# 4. Prompt 템플릿 생성
example_prompt = PromptTemplate.from_template(
    """
    Question:
    {question}
    Answer:
    {answer}
    """
)

# 5. FewShotPromptTemplate 설정
prompt = FewShotPromptTemplate(
    examples=examples,  # 예제 데이터
    example_prompt=example_prompt,  # 예제 출력 형식
    suffix="Question:\n{question}\nAnswer:",  # 사용자 질문과 답변 형식
    input_variables=["question"],  # 입력 변수
)

# 6. 체인 생성
chain = prompt | llm

# 질문에 대한 응답 생성 및 출력
response = chain.invoke(input={"question": "Google이 창립된 연도에 Bill Gates의 나이는 몇 살인가요?"})

# 응답 내용 출력
print(response.content)

print("=" * 80)

# content를 제외한 나머지 메타데이터 출력
print({key: value for key, value in response.__dict__.items() if key != 'content'})
