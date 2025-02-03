import streamlit as st

st.title("계산기")

if 'ceremony' not in st.session_state:
    st.session_state.ceremony = ""

calculation = st.text_input("계산식을 입력해주세요 (+, -, *, /):", value=st.session_state.ceremony, key="input_field")
st.session_state.ceremony = calculation

if st.button("계산"):
    try:
        result = eval(st.session_state.ceremony)
        st.success(f"결과: {result}")
    except Exception as e:
        st.error(f"오류: {e}")
