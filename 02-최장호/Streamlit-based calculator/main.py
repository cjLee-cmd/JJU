import streamlit as st

st.title("Streamlit을 활용한 계산기")


if "result" not in st.session_state:
    st.session_state.result = "0"
if "current" not in st.session_state:
    st.session_state.current = ""


def calculate(expression):
    try:
        return str(eval(expression))
    except ZeroDivisionError:
        return "Zero 처리 할 수 없습니다!"
    except Exception:
        return "에러 발생!"


def create_buttons():
    buttons = [
        ["7", "8", "9", "/"],
        ["4", "5", "6", "*"],
        ["1", "2", "3", "-"],
        ["C", "0", "=", "+"]
    ]
    
    for row in buttons:
        cols = st.columns(4)  
        for i, btn in enumerate(row):
            if cols[i].button(btn):
                handle_click(btn)


def handle_click(btn):
    if btn == "C":  
        st.session_state.current = ""
        st.session_state.result = "0"
    elif btn == "=":  
        st.session_state.result = calculate(st.session_state.current)
        st.session_state.current = ""
    else: 
        st.session_state.current += btn


create_buttons()
st.write(f"### 입력된 숫자: `{st.session_state.current}`")
st.write(f"### 연산 결과: `{st.session_state.result}`")
