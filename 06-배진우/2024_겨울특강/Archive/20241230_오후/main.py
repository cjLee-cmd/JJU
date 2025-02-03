import streamlit as st
import google.generativeai as genai
import os

# 페이지 설정
st.set_page_config(
    page_title="Gemini 맞춤법 검사기",
    page_icon="✨",
)

st.title("Gemini 맞춤법 검사기 ✨")
st.write("Gemini Pro 모델을 사용하여 맞춤법 및 문법 오류를 수정합니다.")

# API 키 설정 (환경 변수에서 가져오기)
GOOGLE_API_KEY = os.getenv("")
if not GOOGLE_API_KEY:
    st.error("API 키를 환경 변수 GOOGLE_API_KEY에 설정해주세요.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def check_spelling(text):
    """
    Gemini Pro 모델을 사용하여 입력된 텍스트의 맞춤법 및 문법 오류를 수정합니다.

    Args:
        text: 맞춤법 검사를 할 텍스트

    Returns:
        수정된 텍스트
    """
    prompt_parts = [
        "다음 문장의 맞춤법과 문법 오류를 수정해 주세요:",
        text
    ]
    response = model.generate_content(prompt_parts)
    return response.text

# 사용자 입력 받기
text_to_check = st.text_area("맞춤법 검사를 할 문장을 입력하세요:", height=200)

if st.button("맞춤법 검사 실행"):
    if text_to_check:
        with st.spinner('열심히 교정 중...'):
            corrected_text = check_spelling(text_to_check)
        st.subheader("수정된 문장:")
        st.write(corrected_text)
    else:
        st.warning("검사할 문장을 입력해주세요.")