import streamlit as st
from praisonaiagents import Agent, Agents, Tools

# Streamlit UI 설정
st.title("AI 에이전트 상호작용")
topic = st.text_input("조사할 주제를 입력하세요:", "인공지능과 딥러닝")

# 연구 에이전트: 주제에 대해 인터넷에서 정보 검색
research_agent = Agent(
    instructions=f'너는 {topic}에 관해서 알려주는 agent야.',
    tools=[Tools.internet_search],
    llm='gpt-4o'
)

# 요약 에이전트: 검색 결과를 요약하고 한국어로 설명
summarize_agent = Agent(
    instructions='너는 요약 agent로 관련된 정보를 찾아서 설명해주는 agent야.',
    llm='gpt-4o',
)

# 버튼 클릭 시 작업 실행
if st.button("연구 및 요약 시작"):
    agents = Agents(agents=[research_agent, summarize_agent])
    agents.start()

    # 에이전트들의 결과를 화면에 표시
    st.subheader("연구 결과:")
    st.write(research_agent.response)  # 'response'에 에이전트의 출력이 담길 것으로 가정
    
    st.subheader("요약:")
    st.write(summarize_agent.response)  # 'response'에 요약된 내용이 담길 것으로 가정

print("hi")