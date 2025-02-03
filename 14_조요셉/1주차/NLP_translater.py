import streamlit as st
import google.generativeai as genai
import re

# Google Generative AI 설정
GOOGLE_API_KEY = "your api key"  # 여기에 Google API 키 입력
genai.configure(api_key=GOOGLE_API_KEY)

# 섹션별 분석 결과를 포맷팅하는 함수
def parse_and_format_analysis(analysis_text):
    """
    분석 결과 텍스트를 섹션별로 나누어 포맷팅. 줄바꿈 구분이 아닌 섹션 제목 탐지 기반으로 처리.
    """
    sections = {
        "오류 내용 출력": "",
        "영어 문장에서 사용된 문법 규칙": "",
        "의미와 스타일 개선을 위한 제안": "",
        "수정된 한국어 번역": "",
    }

    section_pattern = re.compile(r"\*\*(.+?):\*\*")
    current_section = None

    for line in analysis_text.split("\n"):
        line = line.strip()
        match = section_pattern.match(line)
        if match:
            section_title = match.group(1).strip()
            if section_title in sections:
                current_section = section_title
                sections[current_section] = line[match.end():].strip()
            else:
                current_section = None
        elif current_section:
            sections[current_section] += " " + line.strip()

    for section in sections:
        sections[section] = re.sub(r'["\'*]', '', sections[section]).strip()

    return sections

# Streamlit 앱 UI 설정
st.title("AI 영어 문장 생성 및 번역 분석")
st.write("Google Generative AI (Gemini-Pro)를 사용해 영어 문장을 생성하고 번역을 분석합니다.")

# 영어 문장 생성 함수
def generate_english_sentence():
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Please make an English sentence.")
    return response.text.strip() if response and hasattr(response, 'text') else "문장을 생성할 수 없습니다."

# 초기 세션 상태 설정
if "english_sentence" not in st.session_state:
    st.session_state["english_sentence"] = generate_english_sentence()

# 생성된 영어 문장 표시
st.subheader("1. 생성된 영어 문장")
english_sentence = st.session_state["english_sentence"]
st.info(english_sentence)

# 한국어 번역 입력받기
korean_translation = st.text_area("2. 번역된 한국어 문장을 입력하세요:", placeholder="여기에 번역을 입력하세요...")

# 번역 분석 버튼
if st.button("3. 번역 분석"):
    if not korean_translation:
        st.error("번역된 문장을 입력해주세요!")
    else:
        # 분석 요청 프롬프트 생성
        translated_prompt = f"""
        이 문장을 한국어로 번역하고 분석하세요:
        영어: {english_sentence}
        한국어: {korean_translation}

        아래 양식에 맞추어 영어 문장을 상세히 분석하고 아래 양식과 철저히 똑같이 출력하세요.:
        - 오류 내용 출력(번역된 문장에 오류가 있다면 [의미 오류, 구문 오류, 단어 선택 오류, 어순 오류 ,문화적 오류]중 선택하고 오류 내용 출력) : 
        - 영어 문장에서 사용된 문법 규칙 : 
        - 의미와 스타일 개선을 위한 제안 : 
        - 수정된 한국어 번역 : 
        """

        # Google Generative AI 호출
        with st.spinner("분석 중..."):
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(translated_prompt)
            analysis_result = response.text if response and hasattr(response, 'text') else "응답을 생성할 수 없습니다."

            # 섹션별 결과 파싱 및 출력
            formatted_analysis = parse_and_format_analysis(analysis_result)

            # 결과 표시
            st.subheader("4. 번역 분석 결과")
            for section, content in formatted_analysis.items():
                st.write(f"**{section}**")
                st.info(content)

# 세션 초기화 버튼
if st.button("새 문장 생성"):
    st.session_state["english_sentence"] = generate_english_sentence()
    st.rerun()
