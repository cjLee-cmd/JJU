import os
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from praisonaiagents import Agent, Agents, Tools
from datetime import datetime
from langchain.prompts import PromptTemplate

load_dotenv()

def prompt_to_topic_string(chain_prompt):
    """
    딕셔너리 형식의 chain_prompt를 토픽으로 변환하는 문자열 함수
    """
    return f"{chain_prompt['match_date']}에 열린 {chain_prompt['league']} 경기에서 {chain_prompt['team_a']}와 {chain_prompt['team_b']}의 경기 분석"

def agents(chain_prompt):

    search_topic = prompt_to_topic_string(chain_prompt)

    research_agent = Agent(
        instructions=f"You're a research agent to search recent content about {search_topic} from various sources, including news articles, blogs, cafes, and other websites. Find the 10 most recent pieces of content and summarize them. Answer in Korean.",
        tools=[Tools.internet_search]
    )

    summerarize_Agent = Agent(
        instructions="you'r a summarize agent to summarise research agent's findings. Answer in Korean"
    )

    agents = Agents(agents=[research_agent, summerarize_Agent])
    result = agents.start()
    return result["task_results"][0].raw, result["task_results"][1].raw

def template(news, ai_summary):
    prompt_template = PromptTemplate(
        template = """
    오늘 날짜는 {today}입니다. 아래에 주어진 정보를 참고하여 {match_date}에 열린 {league} 경기에서 {team_a}와 {team_b}의 경기를 분석해 주세요:

    ### **참고 정보**
    - **뉴스 정보**: {news}
    - **AI 요약 정보**: {ai_summary}

    ### **분석 요청 사항**

    1. **경기 요약**: 각 팀의 주요 상황, 경기의 하이라이트(득점, 중요한 판정, 결정적 순간 등)를 설명해주세요.

    2. **팀별 전술 분석**: 
    - 팀 A ({team_a})와 팀 B ({team_b})가 각각 어떤 전술을 사용했는지 설명하고, 경기 중 주요 지표(예: 점유율, 패스 성공률, 슛 시도 위치 등)를 포함해주세요.
    - 각 팀이 어떤 방식으로 공격을 전개했는지, 그리고 방어 시 어떤 전략을 취했는지 설명해주세요.

    3. **주요 선수 퍼포먼스**: 
    - 팀 A와 팀 B에서 각각 가장 눈에 띈 3명의 선수를 선정하고, 각 선수의 역할(득점, 패스, 수비 등)과 주요 지표(득점 수, 어시스트, 인터셉트 횟수 등)를 바탕으로 설명해주세요.

    4. **데이터 시각화 설명**:
    - 경기 중 각 팀의 주요 활동 영역을 설명해주세요. (예: 축구의 경우 패스맵과 히트맵, 농구의 경우 포지션별 이동 경로, 배구의 경우 득점 위치 등)

    5. **비교 분석**:
    - 이 경기 전 두 팀의 최근 5경기 기록(승, 무, 패)을 참고하여 경기 결과와 비교해 주세요. 팀별 경기력 변화에 대한 통찰을 제공해 주세요.

    6. **전략적 개선 제안**:
    - 다음 경기를 위해 팀 A와 팀 B 각각에 대한 개선 방안을 제시해 주세요. 공격 및 수비 측면에서 무엇을 보완해야 할지 설명하고, 새로운 전술적 아이디어를 제안해 주세요.

    """,
        input_variables=["league", "match_date", "team_a", "team_b"],
        partial_variables={"today": datetime.now().strftime("%Y년 %m월 %d일"), "news":news, "ai_summary":ai_summary}
    )
    return prompt_template

llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=0,
)

chain_prompt = {
    "league": "KBO 리그",
    "match_date": "2024년 9월 30일",
    "team_a": "NC",
    "team_b": "KIA"
}

news, ai_summary = agents(chain_prompt)
prompt = template(news, ai_summary)

chain = prompt | llm

result = chain.invoke(chain_prompt).content

def save_results_and_template(chain_prompt, result, prompt_template):
    # 텍스트 파일로 출력 결과 저장
    output_file = f"ANSWERFROMLLM.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    print(f"[디버그] 출력 결과가 {output_file}에 저장되었습니다.")
    
    # YAML 파일로 템플릿 저장
    yaml_file = f"PROMPTTEMPLATE.yaml"
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(prompt_template, f, allow_unicode=True)
    print(f"[디버그] 프롬프트 템플릿이 {yaml_file}에 저장되었습니다.")

# 결과와 템플릿 저장
save_results_and_template(chain_prompt, result, prompt.template)