import json
from loaders import load_file
from splitters import split_texts
from analyzers import generate_embeddings, find_similar_chunks, generate_questions, analyze_with_gpt
from utils import load_env, detect_input_type, detect_file_type

def main(input_data):
    env = load_env()
    if not env.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    
    input_type = detect_input_type(input_data)
    file_type = detect_file_type(input_data) if input_type == "FILE" else "URL"

    # 문서 로드 및 분할
    loader_results = load_file(file_type, input_data)
    split_results = split_texts(file_type, loader_results)

    document_chunks = []
    for chunk in split_results:
        result = chunk.get("result", "")
        if isinstance(result, list):
            document_chunks.extend([item.strip() for item in result if isinstance(item, str)])
        elif isinstance(result, str):
            document_chunks.append(result.strip())

    # Embeddings 생성 (필요 시 저장)
    embeddings = generate_embeddings(document_chunks)

    # 전체 문서를 하나의 문자열로 합침
    document_text = " ".join(document_chunks)
    questions = generate_questions(document_text)
    print("생성된 질문:", questions)

    if not (isinstance(questions, list) and len(questions) == 10):
        print("질문 생성에 문제가 발생했습니다:", questions)
        return

    # 질문 목록을 딕셔너리 형태로 변환 (키를 "질문"으로 통일)
    questions_list = [{"질문": q} for q in questions]
    # analyze_with_gpt 함수를 단 한 번 호출하여 10개의 질문-답변 쌍을 포함하는 JSON 배열을 생성
    qa_pairs = analyze_with_gpt(file_type, questions_list, document_text)
    print("생성된 Q&A 결과:", qa_pairs)

    with open("Result.json", "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)

    print("결과가 'Result.json'에 저장되었습니다.")

if __name__ == "__main__":
    input_data = "/Users/yusangmin/Documents/GitHub/JJU-1/비트컴퓨터_프로젝트_발표/논문1.pdf"
    main(input_data)