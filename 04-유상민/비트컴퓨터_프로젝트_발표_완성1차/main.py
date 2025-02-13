import streamlit as st
import os
import json
import tempfile
import pandas as pd
from loaders import load_file
from splitters.text_splitters import split_texts
from analyzers.gpt_analyzer import generate_questions, evaluate_qa_pairs
from analyzers.embedding_analyzer import generate_embeddings, save_embeddings_to_chromadb, find_similar_chunks

def main():
    st.title("문서 기반 Q&A 시스템")
    st.write(
        "문서를 업로드하면 해당 문서를 분할, 임베딩, ChromaDB 저장 후, "
        "GPT를 통해 10개의 질문을 생성하고, 각 질문에 대해 유사한 문서 조각을 검색하여 답변을 제공합니다. "
        "또한, 생성된 Q&A 결과를 평가하여 총점 100점 만점의 평가 결과를 UI에 표 형식으로 표시합니다."
    )
    
    uploaded_file = st.file_uploader("문서를 업로드하세요", type=["pdf", "hwp", "json", "txt"])
    
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[-1].lower()
        
        # 파일 유형 결정
        if file_extension == ".pdf":
            file_type = "PDF"
        elif file_extension == ".hwp":
            file_type = "HWP"
        elif file_extension == ".json":
            file_type = "JSON"
        elif file_extension == ".txt":
            file_type = "Text"
        else:
            st.error("지원하지 않는 파일 형식입니다.")
            return
        
        st.success(f"파일 업로드 완료: {file_name} ({file_type})")
        
        with st.spinner("문서 내용을 로딩 중입니다..."):
            # 업로드된 파일을 임시 파일로 저장하고, 파일 경로를 load_file에 전달
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            document_content = load_file(file_type, tmp_path)
        
        if document_content:
            st.subheader("문서 미리보기")
            st.write(document_content[:1000] + "...")
        else:
            st.error("문서 내용을 로딩하지 못했습니다.")
            return
        
        if st.button("문서 처리 시작"):
            # Step 1: 문서 분할
            with st.spinner("문서를 분할 중입니다..."):
                split_results = split_texts(file_type, document_content)
            
            # 분할 결과 중 'RecursiveCharacterTextSplitter'를 우선 선택
            selected_chunks = None
            for result in split_results:
                if result.get("splitter") == "RecursiveCharacterTextSplitter":
                    selected_chunks = result.get("result")
                    break
            if not selected_chunks and split_results:
                selected_chunks = split_results[0].get("result")
            
            if not selected_chunks:
                st.error("문서 분할에 실패했습니다.")
                return
            
            st.success(f"문서 분할 완료! 총 {len(selected_chunks)}개의 조각이 생성되었습니다.")
            
            # Step 2: 임베딩 생성 및 ChromaDB 저장
            with st.spinner("임베딩 생성 및 ChromaDB 저장 중입니다..."):
                embeddings = generate_embeddings(selected_chunks)
                save_embeddings_to_chromadb(embeddings, selected_chunks)
            st.success("임베딩 생성 및 ChromaDB 저장 완료!")
            
            # Step 3: GPT를 통한 질문 생성 (10개)
            with st.spinner("질문 생성 중입니다..."):
                questions = generate_questions(document_content)
            
            if isinstance(questions, list) and len(questions) == 10:
                st.subheader("생성된 질문")
                for idx, q in enumerate(questions, start=1):
                    st.write(f"{idx}. {q}")
            else:
                st.error("질문 생성에 문제가 발생했습니다.")
                st.write(questions)
                return
            
            # Step 4: 각 질문에 대해 ChromaDB에서 유사한 문서 조각 검색 (답변 제공)
            qa_results = []
            with st.spinner("질문에 대한 답변 검색 중입니다..."):
                for q in questions:
                    similar_chunks = find_similar_chunks(q, top_k=3)
                    # 유사한 문서 조각들을 하나의 문자열로 결합 (유사도 점수 포함)
                    answer = "\n".join([f"유사도: {sim:.4f}\n{text}" for text, sim in similar_chunks])
                    qa_results.append({"질문": q, "답변": answer})
            
            st.subheader("Q&A 결과")
            for item in qa_results:
                st.markdown(f"**질문:** {item['질문']}")
                st.markdown(f"**답변:** {item['답변']}")
                st.markdown("---")
            
            # Step 5: 평가 진행 (총점 100점)
            with st.spinner("평가 진행 중입니다..."):
                evaluation = evaluate_qa_pairs(qa_results)
            
            st.subheader("평가 결과")
            # 기존 마크다운 출력
            if "evaluations" in evaluation:
                for eval_item in evaluation["evaluations"]:
                    st.markdown(f"**질문:** {eval_item.get('질문', 'N/A')}")
                    st.markdown(f"**답변:** {eval_item.get('답변', 'N/A')}")
                    st.markdown(f"**관련성:** {eval_item.get('관련성', 'N/A')} / 100")
                    st.markdown(f"**정확성:** {eval_item.get('정확성', 'N/A')} / 100")
                    st.markdown(f"**완전성:** {eval_item.get('완전성', 'N/A')} / 100")
                    st.markdown(f"**명료성:** {eval_item.get('명료성', 'N/A')} / 100")
                    st.markdown(f"**점수:** {eval_item.get('score', 'N/A')} / 100")
                    st.markdown("---")
            if "overall_score" in evaluation:
                st.markdown(f"**전체 평균 점수:** {evaluation['overall_score']} / 100")
            
            # 추가: 평가 결과를 Pandas DataFrame으로 UI에 표 형식으로 출력
            if "evaluations" in evaluation and evaluation["evaluations"]:
                eval_df = pd.DataFrame(evaluation["evaluations"])
                st.subheader("평가 결과표")
                st.table(eval_df)
            
            # 최종 결과를 JSON 파일로 다운로드
            final_result = {
                "document_content": document_content,
                "split_chunks": selected_chunks,
                "questions": questions,
                "qa_results": qa_results,
                "evaluation": evaluation
            }
            json_str = json.dumps(final_result, indent=4, ensure_ascii=False)
            st.download_button("최종 결과 JSON 다운로드", data=json_str, file_name="Result.json", mime="application/json")

if __name__ == "__main__":
    main()