import os
import yaml
from dotenv import load_dotenv
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 1. 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. ChatOpenAI 객체 생성
llm = ChatOpenAI(
    temperature=0,  # 창의성 제어
    model_name="gpt-4o",  # 사용할 모델 이름
    api_key=OPENAI_API_KEY,  # OpenAI API Key
)

# 3. Examples 데이터 로드 (YAML 파일 사용)
with open(r"10_여명구\02_Prompt\data\examples.yaml", "r", encoding="utf-8") as f:
    yaml_data = yaml.safe_load(f)  # YAML 파일 로드
    examples = yaml_data["examples"]

# 4. Prompt 템플릿 생성
example_prompt = PromptTemplate.from_template(
    """
    Question:
    {question}
    Answer:
    {answer}
    """
)

# 5. SemanticSimilarityExampleSelector 설정
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,  # 예제 목록
    embeddings=OpenAIEmbeddings(),  # 임베딩 생성 클래스
    vectorstore_cls=Chroma,  # VectorStore 클래스
    k=1,  # 선택할 예제 개수
)

# 6. FewShotPromptTemplate 설정
prompt = FewShotPromptTemplate(
    example_selector=example_selector,  # 선택된 예제들
    example_prompt=example_prompt,  # 예제 출력 형식
    suffix="Question:\n{question}\nAnswer:",  # 사용자 질문과 답변 형식
    input_variables=["question"],  # 입력 변수
)

# 7. 입력 데이터
question = "Google이 창립된 연도에 Bill Gates의 나이는 몇 살인가요?"

# 8. 가장 유사한 예제 선택
selected_examples = example_selector.select_examples({"question": question})
print(f"입력에 가장 유사한 예시:\n{question}\n")
for example in selected_examples:
    print(f'question:\n{example["question"]}')
    print(f'answer:\n{example["answer"]}')

# 9. 선택된 예제 기반으로 프롬프트 생성
example_selector_prompt = prompt.format(question=question)

# 10. 체인 생성 및 실행
chain = prompt | llm
response = chain.invoke({"question": question})

# 11. 결과 출력
print("질문 : ", question)
print("\n")
print("응답 내용:")
print(response.content)

print("\n응답 메타데이터:")
print({key: value for key, value in response.__dict__.items() if key != 'content'})
