import streamlit as st
import tempfile
import os
import json
from langchain.document_loaders import PyPDFLoader
from analyzers.gpt_analyzer import generate_questions, analyze_with_gpt

def extract_text_from_pdf(file) -> str:
    """
    업로드된 PDF 파일 객체(file)를 받아 텍스트를 추출하는 함수입니다.
    PyPDFLoader를 사용하여 각 페이지의 텍스트를 합쳐 하나의 문자열로 반환합니다.
    """
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    # PyPDFLoader를 이용해 PDF 텍스트 추출
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.remove(tmp_path)  # 임시 파일 삭제

    # Document 객체들의 page_content를 하나의 문자열로 결합
    text = " ".join([doc.page_content for doc in docs])
    return text

def main():
    st.title("PDF Q&A 생성기")
    st.write("PDF 파일을 업로드하면 해당 문서 내용을 기반으로 10개의 질문과 답변 쌍을 생성합니다.")

    # PDF 파일 업로드
    uploaded_file = st.file_uploader("PDF 파일을 선택하세요", type="pdf")
    
    if uploaded_file is not None:
        st.info("파일이 업로드되었습니다. 텍스트를 추출 중입니다...")
        document_text = extract_text_from_pdf(uploaded_file)
        
        if document_text:
            st.subheader("추출된 문서 일부")
            st.write(document_text[:1000] + " ...")  # 미리보기 (앞 1000자)
        else:
            st.error("문서에서 텍스트를 추출하지 못했습니다.")
            return

        # 버튼을 눌러 질문 및 답변 생성 실행
        if st.button("질문 및 답변 생성"):
            with st.spinner("GPT를 호출하여 질문을 생성하는 중..."):
                questions = generate_questions(document_text)
            st.subheader("생성된 질문")
            st.write(questions)

            if not (isinstance(questions, list) and len(questions) == 10):
                st.error("질문 생성에 문제가 발생했습니다.")
                return

            # 분석 모듈에서 일관된 키("질문")를 사용하므로 질문 목록을 변환
            questions_list = [{"질문": q} for q in questions]

            with st.spinner("GPT를 호출하여 답변을 생성하는 중..."):
                qa_pairs = analyze_with_gpt("PDF", questions_list, document_text)
            st.subheader("생성된 Q&A 결과")
            st.json(qa_pairs)
            
            # JSON 결과 파일 다운로드 버튼 생성
            json_str = json.dumps(qa_pairs, indent=4, ensure_ascii=False)
            st.download_button("Result.json 다운로드", data=json_str,
                               file_name="Result.json", mime="application/json")

if __name__ == "__main__":
    main()