import os
import csv
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from process_pdf import process_pdf
from vector_database import create_vector_database, query_database
from response_generator import generate_response

def check_credentials(username, password, csv_path="user_credentials.csv"):
    """
    CSV 파일에서 username, password가 일치하는 계정이 있는지 확인.
    일치하면 True, 없으면 False 반환.
    """
    try:
        df = pd.read_csv(csv_path)
        user_row = df[(df["username"] == username) & (df["password"] == password)]
        if not user_row.empty:
            return True
        else:
            return False
    except Exception as e:
        print(f"[Error] Failed to read credentials CSV: {e}")
        return False

def log_user_query(username, question, answer, log_csv_path="user_query_logs.csv"):
    """
    사용자 질의(질문, 답변)를 CSV 파일에 로그로 남김.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp, username, question, answer]
    try:
        with open(log_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print(f"[Error] Failed to write to query log CSV: {e}")

def main_streamlit():
    load_dotenv()

    st.set_page_config(
        page_title="Semantic Analysis",
        page_icon="🔬",
        layout="wide"
    )

    # 세션 스테이트 초기화
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "global_db" not in st.session_state:
        st.session_state["global_db"] = None
    if "texts" not in st.session_state:
        st.session_state["texts"] = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "last_uploaded_filename" not in st.session_state:
        st.session_state["last_uploaded_filename"] = None
    if "editing_index" not in st.session_state:
        st.session_state["editing_index"] = None  # 수정 중인 질문의 인덱스

    # 사이드바 - 로그인 섹션
    st.sidebar.title("User Login")

    if not st.session_state["logged_in"]:
        # 로그인 폼
        username_input = st.sidebar.text_input("Username")
        password_input = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            if check_credentials(username_input, password_input, "user_credentials.csv"):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username_input
                st.sidebar.success(f"로그인 성공: {username_input}")
            else:
                st.sidebar.error("로그인 실패! 아이디 또는 비밀번호를 확인해주세요.")
    else:
        # 로그인된 상태 표시
        st.sidebar.success(f"로그인됨: {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            # 원하면 세션 전체를 초기화
            # st.session_state.clear()
            st.sidebar.info("로그아웃되었습니다.")

    # 로그인된 상태에서만 PDF 업로드, 질의응답 기능 사용 가능
    if st.session_state["logged_in"]:
        # 상단 레이아웃
        st.markdown("<h3 style='text-align:right;'>Jeonju-University</h3>", unsafe_allow_html=True)
        st.title("🔬 Semantic Analysis with LangChain")
        st.markdown("""
        This application allows you to upload a PDF, analyze its content, 
        and retrieve information using natural language queries.

        **Features:**
        - Semantic chunking for better document understanding.
        - Vector database for fast and accurate search.
        - GPT-like streaming responses in Korean.
        """)

        # 사이드바: PDF 업로드
        st.sidebar.subheader("PDF Upload")
        uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

        # PDF 업로드 -> 텍스트 추출 -> 벡터 DB 생성
        if uploaded_file:
            if st.session_state["last_uploaded_filename"] != uploaded_file.name:
                st.session_state["last_uploaded_filename"] = uploaded_file.name

                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.read())

                # PDF 처리
                try:
                    st.session_state["texts"] = process_pdf(temp_file_path)
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

                # 벡터 DB 생성
                try:
                    st.session_state["global_db"] = create_vector_database(st.session_state["texts"])
                except Exception as e:
                    st.error(f"Error creating vector DB: {e}")

        # 기존 메시지 출력
        for i, msg in enumerate(st.session_state["messages"]):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
            # 사용자 메시지의 경우 "수정" 버튼
            if msg["role"] == "user":
                if st.button("수정", key=f"edit-{i}"):
                    st.session_state["editing_index"] = i

        # 수정 중인 질문 처리
        if st.session_state["editing_index"] is not None:
            editing_index = st.session_state["editing_index"]
            old_question = st.session_state["messages"][editing_index]["content"]

            # 수정용 입력창
            new_question = st.text_input("질문 수정:", value=old_question, key="edit_input")
            if st.button("수정 완료"):
                # 수정된 질문 저장
                st.session_state["messages"][editing_index]["content"] = new_question

                # 바로 다음 메시지가 답변이면 제거 (수정된 질문에 새 답변을 생성하기 위해)
                if len(st.session_state["messages"]) > editing_index + 1:
                    if st.session_state["messages"][editing_index + 1]["role"] == "assistant":
                        del st.session_state["messages"][editing_index + 1]

                # 수정된 질문에 대한 새로운 답변 생성
                if st.session_state["global_db"] is not None:
                    with st.chat_message("assistant"):
                        with st.spinner("답변 생성 중..."):
                            mmr_docs = query_database(st.session_state["global_db"], new_question)
                            new_answer = generate_response(new_question, mmr_docs)

                    # 새로운 답변 저장
                    st.session_state["messages"].insert(editing_index + 1, {
                        "role": "assistant",
                        "content": new_answer
                    })

                    # 로그 기록 (수정된 질문 기준)
                    log_user_query(st.session_state["username"], new_question, new_answer)

                # 수정 상태 해제
                st.session_state["editing_index"] = None
                st.success("질문이 수정되고, 새로운 답변이 생성되었습니다!")
        else:
            # 새 질문 입력창
            user_input = st.chat_input("Enter your question...")
            if user_input:
                # 사용자 메시지 추가
                st.session_state["messages"].append({
                    "role": "user",
                    "content": user_input
                })
                with st.chat_message("user"):
                    st.write(user_input)

                # 벡터 DB가 아직 없으면 경고
                if st.session_state["global_db"] is None:
                    with st.chat_message("assistant"):
                        st.warning("Please upload a PDF file first to create a database.")
                else:
                    try:
                        mmr_docs = query_database(st.session_state["global_db"], user_input)
                    except Exception as e:
                        with st.chat_message("assistant"):
                            st.error(f"An error occurred during querying: {e}")
                        return

                    # 답변 생성
                    with st.chat_message("assistant"):
                        with st.spinner("답변 생성 중..."):
                            partial_text = generate_response(user_input, mmr_docs)

                    # 최종 답변을 세션에 추가
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": partial_text
                    })

                    # 질의 로그 남기기
                    log_user_query(st.session_state["username"], user_input, partial_text)
    else:
        # 로그인되지 않은 상태
        st.warning("로그인 후 이용 가능합니다.")

if __name__ == "__main__":
    # pysqlite3를 sqlite3로 교체 (필요 시)
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    import sqlite3

    main_streamlit()
