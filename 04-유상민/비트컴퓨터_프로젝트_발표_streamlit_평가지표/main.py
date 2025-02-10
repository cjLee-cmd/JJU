import streamlit as st
import tempfile
import os
import json
from langchain.document_loaders import PyPDFLoader
from analyzers.gpt_analyzer import generate_questions, analyze_with_gpt, evaluate_qa_pairs

def extract_text_from_pdf(file) -> str:
    """
    업로드된 PDF 파일 객체(file)를 받아 텍스트를 추출하는 함수입니다.
    PyPDFLoader를 사용하여 각 페이지의 텍스트를 합쳐 하나의 문자열로 반환합니다.
    """
    # 업로드된 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    # PyPDFLoader를 사용하여 PDF 파일에서 텍스트 추출
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.remove(tmp_path)  # 임시 파일 삭제

    # 각 Document 객체의 page_content를 결합하여 전체 텍스트 반환
    text = " ".join([doc.page_content for doc in docs])
    return text

def main():
    st.title("PDF 기반 Q&A 생성 및 평가")
    st.write("PDF 파일을 업로드하면, 문서의 전체 내용을 기반으로 10개의 질문과 답변을 생성하고, "
             "각 Q&A 쌍을 관련성, 정확성, 완전성, 명료성 기준으로 0-100점 사이로 평가한 결과와 전체 평균 점수를 제공합니다.")

    # PDF 파일 업로드
    uploaded_file = st.file_uploader("PDF 파일을 선택하세요", type="pdf")
    
    if uploaded_file is not None:
        st.info("파일 업로드 완료. 텍스트를 추출 중입니다...")
        document_text = extract_text_from_pdf(uploaded_file)
        
        if document_text:
            st.subheader("추출된 문서 일부")
            st.write(document_text[:1000] + " ...")  # 미리보기: 앞 1000자
        else:
            st.error("문서에서 텍스트를 추출하지 못했습니다.")
            return

        # 버튼을 누르면 Q&A 생성 및 평가 실행
        if st.button("질문, 답변 생성 및 평가 실행"):
            # 1. 질문 생성
            with st.spinner("질문 생성 중..."):
                questions = generate_questions(document_text)
            st.subheader("생성된 질문")
            st.write(questions)
            if not (isinstance(questions, list) and len(questions) == 10):
                st.error("질문 생성에 문제가 발생했습니다.")
                return

            # 2. Q&A 생성 (질문 목록은 {"질문": 질문내용} 형식으로 통일)
            questions_list = [{"질문": q} for q in questions]
            with st.spinner("답변 생성 중..."):
                qa_pairs = analyze_with_gpt("PDF", questions_list, document_text)
            st.subheader("생성된 Q&A 쌍")
            st.json(qa_pairs)

            # 3. 평가 수행 (각 Q&A 쌍에 대해 0-100점 평가 및 전체 평균 점수 계산)
            with st.spinner("평가 진행 중..."):
                evaluation = evaluate_qa_pairs(qa_pairs)
            st.subheader("평가 결과")
            st.json(evaluation)

            # 4. 세부 평가지표 표시
            if "evaluations" in evaluation:
                st.markdown("### 세부 평가 지표")
                for eval_item in evaluation["evaluations"]:
                    st.markdown(f"**질문:** {eval_item['질문']}")
                    st.markdown(f"**답변:** {eval_item['답변']}")
                    st.markdown(f"**점수:** {eval_item['score']}")
                    st.markdown("---")
            if "overall_score" in evaluation:
                st.markdown(f"**전체 평가 점수:** {evaluation['overall_score']}")

            # 최종 결과를 JSON 파일로 다운로드할 수 있도록 설정
            final_result = {"qa_pairs": qa_pairs, "evaluation": evaluation}
            json_str = json.dumps(final_result, indent=4, ensure_ascii=False)
            st.download_button("Result.json 다운로드", data=json_str, file_name="Result.json", mime="application/json")

if __name__ == "__main__":
    main()