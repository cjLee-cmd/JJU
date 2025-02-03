from praisonaiagents import Agent, Agents, Tools

 # 동적 주제 입력
topic = input("조사할 주제를 입력하세요: ")

# Research agent: 인터넷에서 인공지능 및 딥러닝 관련 정보를 검색
research_agent = Agent(
    instructions=f'너는 {topic}에 관해서 알려주는 agent야.',
    tools=[Tools.internet_search],
    llm='gpt-4o'
)

# Summarize agent: 검색 결과를 요약하고 한국어로 설명
summarize_agent = Agent(
    instructions='너는 요약 agent로 관련된 정보를 찾아서 설명해주는 agent야.',
    llm='gpt-4o',
)

# 작업 실행
agents = Agents(agents=[research_agent, summarize_agent])
agents.start()