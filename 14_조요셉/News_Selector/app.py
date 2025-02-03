import openai
import streamlit as st
from praisonaiagents import Agent, Agents, Tools

# Sidebar: API Key and Links
with st.sidebar:
    st.markdown("### 설정")
    "[PraisonAI Docs](https://praison.ai/documentation)"
    "[View the source code](https://github.com/your-repo/news-summarizer)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/your-repo/news-summarizer?quickstart=1)"

# App Title
st.title("📰 뉴스 요약 도우미")
st.caption("🔍 최신 뉴스를 검색하고 요약합니다")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "result" not in st.session_state:
    st.session_state["result"] = None

# Display past messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input for search topic
if prompt := st.chat_input("검색할 주제를 입력하세요:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize agents
    search_topic = prompt
    st.chat_message("assistant").write(f"'{search_topic}'에 대한 뉴스를 검색합니다...")

    research_agent = Agent(
        instructions=f"You're a research agent to search recent news about {search_topic}. Find the 10 most recent news articles and summarize them. Answer in Korean",
        tools=[Tools.internet_search],
    )

    summarize_agent = Agent(
        instructions="You're a summarize agent to summarize research agent's findings. summarize the content in more detail and with more specifics. Answer in Korean",
    )

    agents = Agents(agents=[research_agent, summarize_agent])

    # Run the agents
    with st.spinner("검색 중입니다. 잠시만 기다려 주세요..."):
        result = agents.start()

    # Store and display result
    st.session_state.result = result
    st.session_state.messages.append({"role": "assistant", "content": st.session_state.result["task_results"][0].raw})

# Display final result in a text area
if st.session_state.result:
    st.chat_message("assistant").write(st.session_state.result["task_results"][1].raw, height=150)