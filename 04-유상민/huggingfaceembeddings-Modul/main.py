import json
from loaders import load_file
from splitters import split_texts
from analyzers import generate_embeddings, find_most_similar, generate_questions, analyze_with_gpt
from utils import load_env, detect_input_type, detect_file_type

def main(input_data):
    # 환경 변수 로드
    env = load_env()
    if not env.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

    input_type = detect_input_type(input_data)
    file_type = detect_file_type(input_data) if input_type == "FILE" else "URL"

    print(f"1. Input 데이터 유형: {input_type}")
    print(f"2. 파일 타입: {file_type}")

    try:
        # 문서 로드
        loader_results = load_file(file_type, input_data)
        if not loader_results.strip():
            raise ValueError(f"Loader 결과가 비어 있습니다. file_type: {file_type}, input_data: {input_data}")
        print(f"Loader Results: {loader_results[:100]}")  # 첫 100자 미리보기

        # 텍스트 분리
        split_results = split_texts(file_type, loader_results)
        print("Split Results (Debug):", split_results)

        # Document chunks 생성
        document_chunks = []
        for chunk in split_results:
            result = chunk.get("result", "")
            if isinstance(result, list):  # 리스트 처리
                document_chunks.extend([item.strip() for item in result if isinstance(item, str)])
            elif isinstance(result, str):  # 문자열 처리
                document_chunks.append(result.strip())

        if not document_chunks:
            raise ValueError("Document chunks가 비어 있습니다. Splitter 결과를 확인하세요.")
        print(f"Document Chunks (Preview): {document_chunks[:3]}")

        # Embeddings 생성
        embeddings = generate_embeddings(document_chunks)
        print(f"Embeddings 생성 완료: {embeddings.shape}")

        # 질문 생성
        questions = generate_questions(" ".join(document_chunks))
        print(f"Generated Questions: {questions}")

        # GPT 분석 및 답변 생성
        qa_pairs = []
        for question in questions:
            best_chunk, score = find_most_similar(question, document_chunks, embeddings)
            answer = analyze_with_gpt(file_type, (loader_results, split_results), best_chunk)
            qa_pairs.append({
                "질문": question,
                "답변": answer.strip() if isinstance(answer, str) else "답변 생성 실패",
                "점수": round(float(score), 4)
            })
            print(f"질문: {question}\n답변: {answer}\n유사도 점수: {round(float(score), 4)}\n")

        # 결과 저장
        with open("Result.json", "w") as f:
            json.dump(qa_pairs, f, indent=4, ensure_ascii=False)

        print("결과가 'Result.json'에 저장되었습니다.")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    input_data = "/Users/yusangmin/Documents/GitHub/JJU-1/04-유상민/huggingfaceembeddings-Modul/논문1.pdf"  # 여기에 실제 파일 경로 입력
    main(input_data)
