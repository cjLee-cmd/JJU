import streamlit as st
from praisonaiagents import Agent, Agents, Tools

# Streamlit 애플리케이션 시작
st.title("AI Agents Interaction")

# 동적 주제 입력
topic = st.text_input("조사할 주제를 입력하세요", "")

if topic:
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

    # 결과 저장용 변수
    conversation = []

    with st.spinner("작업 실행 중..."):
        # Agent 시작 및 대화 로그 저장
        def collect_logs(agent, message):
            conversation.append(f"**{agent.name}**: {message}")
            st.write(f"**{agent.name}**: {message}")

        agents.on_message = collect_logs
        agents.start()

    # 대화 로그 출력
    st.header("대화 기록")
    for log in conversation:
        st.write(log)
else:
    st.write("주제를 입력하고 엔터를 눌러주세요!")
