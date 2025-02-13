# qa_evaluation.py

import json, re
from config import OPENAI_API_KEY, MODEL_NAME
from langchain_community.chat_models import ChatOpenAI
import chromadb
from sentence_transformers import SentenceTransformer

# ChatOpenAI 인스턴스 (QA 관련 작업용)
chat_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=MODEL_NAME)

# ChromaDB와 임베딩 모델 초기화 (임베딩 및 검색용)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_questions(document_text: str, num_questions: int = 20) -> list:
    prompt = f"""
다음 문서를 읽고, 그 내용과 관련된 질문 {num_questions}개를 생성해 주세요.
문서:
{document_text}

각 질문은 명확하고 구체적이어야 하며, 문서의 핵심 내용을 반영해야 합니다.
응답은 각 질문을 번호와 함께 줄바꿈으로 구분하여 작성해 주세요.
    """
    response = chat_llm.predict(prompt)
    questions = []
    for line in response.splitlines():
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            cleaned = re.sub(r'^\d+\.\s*', '', line)
            cleaned = cleaned.lstrip("-").strip()
            questions.append(cleaned)
    if len(questions) < num_questions:
        questions = [q.strip() for q in response.splitlines() if q.strip()]
    return questions[:num_questions]

def find_similar_chunks(query: str, top_k: int = 5) -> list:
    collection = chroma_client.get_collection(name="document_embeddings")
    query_embedding = embedding_model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    similar_docs = []
    for idx, doc_id in enumerate(results["ids"][0]):
        doc_text = collection.get([doc_id])["metadatas"][0]["text"]
        similarity_score = results["distances"][0][idx]
        similar_docs.append((doc_text, similarity_score))
    return similar_docs

def generate_answer(question: str, context: str) -> str:
    prompt = f"""
문서 내용:
{context}

위 문서를 참고하여 아래 질문에 대해 자세하고 구체적으로 답변해 주세요.
질문: {question}
답변:
    """
    answer = chat_llm.predict(prompt)
    return answer.strip()

def evaluate_answer(question: str, answer: str) -> dict:
    prompt = f"""
다음 질문과 답변에 대해 평가해 주세요.
질문: {question}
답변: {answer}

평가 항목:
1. 정확성: 답변이 질문에 대해 얼마나 정확한지.
2. 관련성: 답변이 문서 내용과 얼마나 관련이 있는지.
3. 완전성: 답변이 질문의 모든 요소를 포함하는지.
4. 명료성: 답변이 명확하고 이해하기 쉬운지.

각 항목에 대해 0부터 100까지 점수를 부여하고, 최종 점수는 이 항목들의 평균입니다.
응답은 반드시 아래 JSON 형식으로 해 주세요:
{{
    "final_score": <최종 점수>,
    "details": {{
        "accuracy": <정확성 점수>,
        "relevance": <관련성 점수>,
        "completeness": <완전성 점수>,
        "clarity": <명료성 점수>
    }}
}}
    """
    eval_response = chat_llm.predict(prompt)
    try:
        json_str_match = re.search(r'\{.*\}', eval_response, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group()
            eval_result = json.loads(json_str)
            if "final_score" in eval_result and "details" in eval_result:
                return eval_result
        return {"final_score": 50, "details": {"accuracy": 50, "relevance": 50, "completeness": 50, "clarity": 50}}
    except Exception as e:
        print("평가 결과 파싱 중 에러:", e)
        return {"final_score": 50, "details": {"accuracy": 50, "relevance": 50, "completeness": 50, "clarity": 50}}

def qa_pipeline_minimal(document_text: str):
    """
    입력된 문서 전체를 기반으로 20개의 질문을 생성하고,
    각 질문에 대해 답변을 생성 및 평가한 후, 
    최종 결과(질문, 답변, 평가 정보)와 전체 평균 점수를 반환합니다.
    중간 진행 상황은 UI에 출력하지 않습니다.
    """
    questions = generate_questions(document_text, num_questions=20)
    qa_results = []
    total_score = 0
    for q in questions:
        similar_docs = find_similar_chunks(q, top_k=5)
        context = "\n".join([doc for doc, score in similar_docs])
        answer = generate_answer(q, context)
        eval_result = evaluate_answer(q, answer)
        qa_results.append({
            "question": q,
            "answer": answer,
            "evaluation": eval_result
        })
        total_score += eval_result.get("final_score", 50)
    average_score = total_score / len(questions) if questions else 0
    return qa_results, average_score

if __name__ == "__main__":
    # 테스트 실행
    sample_text = "여기에 테스트할 문서 전체 내용 입력"
    results, avg_score = qa_pipeline_minimal(sample_text)
    print("QA 결과:", results)
    print("평균 점수:", avg_score)