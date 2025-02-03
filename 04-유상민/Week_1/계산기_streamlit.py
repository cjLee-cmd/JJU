import streamlit as st

def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        return "0으로 나눌 수 없어요! 💔"
    return x / y

# CSS를 통한 스타일링
st.markdown(
    """
    <style>
    /* 전체 배경 그라데이션 */
    .main {
        background: linear-gradient(to bottom right, #ffe4e1, #fff0f5);
        color: #FF69B4;
        font-family: "Comic Sans MS", cursive, sans-serif;
    }
    /* 제목 색상 */
    h1, h2, h3, h4, h5, h6 {
        color: #FF1493 !important;
    }

    /* 위젯 라벨 색상 */
    label {
        color: #FF69B4 !important;
    }

    /* 버튼 스타일 */
    button[kind="primary"] {
        background-color: #ffb6c1 !important;
        border: 2px solid #ff69b4 !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: bold !important;
    }

    /* 셀렉트박스 화살표 색상 변경 */
    [data-baseweb="select"] .css-1wa3eu0-placeholder{
        color: #ff69b4 !important;
    }
    [data-baseweb="select"] .css-1hwfws3 {
        border-color: #ff69b4 !important;
    }

    /* 숫자 입력 필드 테두리 색상 */
    input[type="number"] {
        border: 2px solid #ff69b4 !important;
        border-radius: 8px !important;
    }

    </style>
    """, unsafe_allow_html=True
)

# 페이지 타이틀
st.title('💕✨ 귀욤뽀짝 계산기 ✨💕')

# 숫자 입력 섹션
st.write("**계산할 숫자를 입력해 주세요!** 🥰")

num1 = st.number_input("첫 번째 숫자:", value=0.0)
operation = st.selectbox("연산자:", ["+", "-", "*", "/"])
num2 = st.number_input("두 번째 숫자:", value=0.0)

# 계산 버튼
if st.button("💖 계산하기 💖"):
    if operation == "+":
        result = add(num1, num2)
    elif operation == "-":
        result = subtract(num1, num2)
    elif operation == "*":
        result = multiply(num1, num2)
    elif operation == "/":
        result = divide(num1, num2)
    
    # 결과 출력
    st.markdown(f"**결과: {result}** 😍")
