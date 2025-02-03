import os
from dotenv import load_dotenv

# 1) .env 파일에서 환경 변수를 먼저 로드
load_dotenv()

# 2) openai.api_key를 미리 설정
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# (선택) API 키가 없는 경우 Streamlit 없이 간단히 에러 처리
if not openai.api_key:
    print("ERROR: OPENAI_API_KEY가 설정되지 않았습니다. .env 또는 환경 변수를 확인하세요.")
    # 필요하다면 sys.exit(1) 등으로 종료할 수 있음
    # import sys
    # sys.exit(1)

# 3) 이제 praisonaiagents와 streamlit을 import
import streamlit as st
from praisonaiagents import Agent, Agents, Tools

def main():
    st.title("PraisionAI Agents Demo")
    st.markdown("이 애플리케이션은 OpenAI API를 사용하여 **검색** 및 **요약**을 수행합니다.")
    
    # 사용자에게 검색할 주제를 입력받습니다.
    topic = st.text_input("조사할 주제를 입력하세요")

    # 버튼을 누르면 검색 및 요약 수행
    if st.button("검색 실행"):
        with st.spinner("에이전트가 정보를 수집 중입니다... 잠시만 기다려주세요."):
            # 1) 인터넷에서 검색할 Agent
            research_agent = Agent(
                instructions=f"너는 {topic}에 관해서 알려주는 agent야.",
                tools=[Tools.internet_search],  # 인터넷 검색 기능
                llm="gpt-4o"
            )

            # 2) 요약을 담당할 Agent
            summarize_agent = Agent(
                instructions="너는 요약 agent로 관련된 정보를 찾아서 설명해주는 agent야.",
                llm="gpt-4o"
            )

            # 여러 Agent를 묶어서 실행
            agents = Agents(agents=[research_agent, summarize_agent])
            result = agents.start()

        st.success("검색이 완료되었습니다!")
        st.write("### 결과 요약")
        st.write(result)

if __name__ == "__main__":
    # 만약 Streamlit UI를 통해도 키가 없을 때 안내하고 싶다면:
    if not openai.api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다. .env 파일 또는 환경 변수를 확인하세요.")
        st.stop()

    main()
