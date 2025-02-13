import streamlit as st
import tempfile
import os
import pandas as pd
import json

from pdf_loaders import load_file
from pdf_splitters import split_content
from evaluate import evaluate_loader_splitter
from config import MODEL_NAME, OPENAI_API_KEY
from embed_best import generate_embeddings, save_embeddings_to_chromadb
from qa_evaluation import qa_pipeline_minimal


st.title("PDF 평가 및 QA 워크플로우")
st.write(
    "PDF 파일을 업로드하면, 로더/스플리터 평가와 QA 평가의 최종 요약 결과만 표시되고, "
    "세부 결과는 다운로드 버튼을 통해 확인할 수 있습니다."
)

# PDF 파일 업로드
uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])

if uploaded_file is not None:
    # 업로드된 파일을 임시 파일로 저장 (경로 필요)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_file_path = tmp.name

    st.info("파일 처리 중입니다. 잠시만 기다려주세요...")


    evaluation_results = []  # 로더/스플리터 평가 결과 저장용
    loader_types = ["PYPDF", "PYMuPDF", "PYPDFium", "PYPDFMiner", "PYPDFPlumber"]
# 각 로더별로 PDF 파일 처리 (중간 과정 출력 없이 내부적으로 처리)
    for loader_type in loader_types:
        try:
            content = load_file(loader_type, tmp_file_path)
            splitters_results = split_content(content)
            for splitter_name, chunks in splitters_results.items():
                if isinstance(chunks, list):
                    joined_text = "\n".join(chunks)
                    eval_result = evaluate_loader_splitter(joined_text)
                    final_score = eval_result.get("final_score", 50)
                    num_chunks = len(chunks)
                    evaluation_results.append({
                        "Loader": loader_type,
                        "Splitter": splitter_name,
                        "Final_Score": final_score,
                        "Num_Chunks": num_chunks,
                        "Details": eval_result,  # 세부 평가 결과 (다운로드용)
                        "Text": joined_text       # 최고 평가 조합 선정 및 임베딩용
                    })
                else:
                    evaluation_results.append({
                        "Loader": loader_type,
                        "Splitter": splitter_name,
                        "Final_Score": 0,
                        "Num_Chunks": 0,
                        "Details": {"error": chunks},
                        "Text": ""
                    })
        except Exception as e:
            st.error(f"로더 '{loader_type}' 처리 중 에러 발생: {e}")

    os.remove(tmp_file_path)

    if evaluation_results:
        # Final_Score 기준 내림차순 정렬하여 최고 평가 조합 선택
        evaluation_results.sort(key=lambda x: x["Final_Score"], reverse=True)
        summary_data = [
            {
                "Loader": res["Loader"],
                "Splitter": res["Splitter"],
                "Final_Score": res["Final_Score"],
                "Num_Chunks": res["Num_Chunks"]
            }
            for res in evaluation_results
        ]
        summary_df = pd.DataFrame(summary_data)
        st.subheader("최종 평가 결과 (요약)")
        st.dataframe(summary_df.reset_index(drop=True))
        
        # 다운로드: 로더-Splitter 평가 결과 요약 (Loader, Splitter, Final_Score만 포함)
        ls_download_data = [
            {
                "Loader": res["Loader"],
                "Splitter": res["Splitter"],
                "Final_Score": res["Final_Score"]
            }
            for res in evaluation_results
        ]
        ls_json = json.dumps(ls_download_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="Loader-Splitter 평가 결과 다운로드",
            data=ls_json,
            file_name="loader_splitter_summary.json",
            mime="application/json"
        )
        
        # 최고 평가 조합 선택 (가장 높은 Final_Score)
        best_result = evaluation_results[0]
        st.success("최고 평가 조합 선택 완료")
        st.write(f"**Loader:** {best_result['Loader']}")
        st.write(f"**Splitter:** {best_result['Splitter']}")
        st.write(f"**평가 점수:** {best_result['Final_Score']}/100")
        
        best_text = best_result["Text"]
        if best_text:
            # 최고 평가 조합 임베딩 및 ChromaDB 저장
            embeddings = generate_embeddings([best_text])
            save_embeddings_to_chromadb(embeddings, [best_text])
            st.success("최고 평가 조합 임베딩 및 ChromaDB 저장 완료")
            
            # QA 평가 진행: qa_pipeline_minimal 함수는 중간 과정 출력 없이 최종 결과만 반환
            qa_results, average_score = qa_pipeline_minimal(best_text)
            # QA 평가 결과 요약 테이블: 질문, 답변, 최종 점수만 표시
            qa_summary = [
                {
                    "Question": qa["question"],
                    "Answer": qa["answer"],
                    "Final_Score": qa["evaluation"].get("final_score", 50)
                }
                for qa in qa_results
            ]
            qa_df = pd.DataFrame(qa_summary)
            st.subheader("QA 평가 결과 (요약)")
            st.dataframe(qa_df)
            
            # 다운로드: QA 평가 결과 - 질문과 답변만 포함한 파일
            qa_qa_only = [
                {"question": qa["question"], "answer": qa["answer"]}
                for qa in qa_results
            ]
            qa_only_json = json.dumps(qa_qa_only, indent=2, ensure_ascii=False)
            st.download_button(
                label="QA 질문/답변만 다운로드",
                data=qa_only_json,
                file_name="qa_questions_answers.json",
                mime="application/json"
            )
            
            # 다운로드: QA 평가 결과 - 질문, 답변, 세부 평가 포함
            qa_full_data = [
                {"question": qa["question"], "answer": qa["answer"], "evaluation": qa["evaluation"]}
                for qa in qa_results
            ]
            qa_full_json = json.dumps(qa_full_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="QA 평가 세부 결과 다운로드",
                data=qa_full_json,
                file_name="qa_full_evaluation.json",
                mime="application/json"
            )
            
            st.write(f"전체 평균 QA 점수: {average_score:.2f}")
        else:
            st.error("최고 평가 조합의 텍스트가 비어있어 임베딩 및 QA 평가를 진행할 수 없습니다.")
    else:
        st.error("평가 결과가 없습니다.")


# 사이드바에 프로젝트 정보 표시
st.sidebar.markdown("# 📚 프로젝트 정보")
st.sidebar.markdown("---")

# 개발자 정보 표시 (이모티콘과 함께)
st.sidebar.markdown("## 👨‍💻 개발자 정보")
st.sidebar.markdown("- **유상민** : dbtkd1102@gmail.com")
st.sidebar.markdown("- **신지수** : s01084228436@gmail.com")

# 모델 정보 표시 (이모티콘과 추가 설명 포함)
st.sidebar.markdown("## 🤖 모델 정보")
st.sidebar.markdown(f"- **모델**: {MODEL_NAME} (temperature:0)")
