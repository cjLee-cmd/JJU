# qa_evaluator.py
import json
import re
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOpenAI
from config import OPENAI_API_KEY, MODEL_NAME

# ChatOpenAI 초기화 (질문 생성 및 답변 평가에 사용)
chat_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=MODEL_NAME)

# ChromaDB 및 임베딩 모델 초기화 (답변 검색에 사용)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_questions(document_text, num_questions=20):
    """
    주어진 문서 텍스트를 바탕으로 20개의 질문을 생성합니다.
    """
    prompt = f"""
아래 문서를 읽고, 해당 문서 내용과 관련된 상세한 질문 {num_questions}개를 번호 매겨서 생성해주세요.

문서:
{document_text}

질문:
1.
"""
    response = chat_llm.predict(prompt)
    # 번호 매겨진 질문 형식에서 질문들을 추출합니다.
    questions = []
    for line in response.splitlines():
        line = line.strip()
        if re.match(r'^\d+[\).]', line):
            # 번호와 구분 기호 제거
            q = re.sub(r'^\d+[\).]\s*', '', line)
            questions.append(q)
    # 질문 개수가 부족하면 간단히 줄 단위로 추가
    if len(questions) < num_questions:
        extra = [q for q in response.splitlines() if len(q) > 10]
        questions = extra[:num_questions]
    return questions


def find_similar_chunks(query, top_k=1):
    """
    ChromaDB 컬렉션에서 주어진 질문(query)과 가장 유사한 문서 조각을 검색합니다.
    """
    collection = chroma_client.get_collection(name="document_embeddings")
    query_embedding = embedding_model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    similar_docs = []
    for idx, doc_id in enumerate(results["ids"][0]):
        # 각 문서 조각의 텍스트를 메타데이터에서 추출합니다.
        doc_text = collection.get([doc_id])["metadatas"][0]["text"]
        similarity_score = results["distances"][0][idx]
        similar_docs.append((doc_text, similarity_score))
    return similar_docs


def answer_question(question):
    """
    질문에 대해 ChromaDB에서 가장 유사한 문서 조각(답변)을 반환합니다.
    """
    similar_docs = find_similar_chunks(question, top_k=1)
    if similar_docs:
        answer_text = similar_docs[0][0]
        return answer_text
    else:
        return "관련된 답변을 찾을 수 없습니다."


def evaluate_answer(question, answer, document_text):
    """
    주어진 질문과 그에 대한 답변, 그리고 문서 전체를 바탕으로 답변의 정확성을 평가합니다.
    
    평가 기준:
    1. Accuracy (정확성)
    2. Relevance (관련성)
    3. Completeness (완전성)
    4. Clarity (명료성)
    
    아래와 같은 JSON 형식으로 응답해 주세요:
    {
        "accuracy": <점수>,
        "relevance": <점수>,
        "completeness": <점수>,
        "clarity": <점수>,
        "total_score": <평균 점수>
    }
    """
    eval_prompt = f"""
다음 질문과 답변, 그리고 문서 전체를 바탕으로 답변의 품질을 평가해주세요.

질문: {question}
답변: {answer}
문서 전체:
{document_text}

평가 기준:
1. Accuracy: 답변이 문서 내용과 얼마나 정확하게 일치하는가.
2. Relevance: 답변이 질문에 얼마나 관련 있는가.
3. Completeness: 답변이 질문에 대한 모든 중요한 정보를 포함하는가.
4. Clarity: 답변이 명확하고 이해하기 쉬운가.

각 항목은 0부터 100까지의 점수로 평가하고, 평균 점수를 total_score로 산출해주세요.

응답은 반드시 아래 JSON 형식으로 해주세요:
{{
    "accuracy": <점수>,
    "relevance": <점수>,
    "completeness": <점수>,
    "clarity": <점수>,
    "total_score": <평균 점수>
}}
"""
    response = chat_llm.predict(eval_prompt)
    try:
        json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
        eval_result = json.loads(json_str)
        return eval_result
    except Exception as e:
        print("평가 응답 파싱 중 에러:", e)
        return {
            "accuracy": 50,
            "relevance": 50,
            "completeness": 50,
            "clarity": 50,
            "total_score": 50
        }


def main(document_text):
    # 1. 문서에서 질문 20개 생성
    questions = generate_questions(document_text, num_questions=20)
    evaluations = []
    print("생성된 질문:")
    for i, q in enumerate(questions, start=1):
        print(f"{i}. {q}")
    print("\n--- 답변 생성 및 평가 시작 ---\n")
    
    # 2. 각 질문에 대해 답변을 검색하고 평가
    for q in questions:
        answer = answer_question(q)
        eval_result = evaluate_answer(q, answer, document_text)
        evaluations.append({
            "question": q,
            "answer": answer,
            "evaluation": eval_result
        })
    
    # 3. 각 질문별 평가 결과 출력 및 전체 평균 점수 계산
    total_scores = [e["evaluation"].get("total_score", 50) for e in evaluations]
    overall_score = sum(total_scores) / len(total_scores) if total_scores else 0

    print("각 질문별 평가 결과:")
    for e in evaluations:
        print(f"질문: {e['question']}")
        print(f"답변: {e['answer']}")
        print(f"평가: {e['evaluation']}")
        print("-" * 40)
    print(f"\n전체 평균 점수: {overall_score}")

    return evaluations, overall_score


if __name__ == "__main__":
    # 테스트용: best_result_text 또는 저장된 문서 전체 텍스트를 입력합니다.
    document_text = "여기에 평가할 문서의 전체 텍스트를 입력하세요. (예시 문서 내용)"
    evaluations, overall_score = main(document_text)