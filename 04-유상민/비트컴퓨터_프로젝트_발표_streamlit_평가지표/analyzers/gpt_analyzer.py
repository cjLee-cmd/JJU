import json
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from openai import OpenAIError
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# .env 파일에서 환경 변수 로드
load_dotenv()

# 임베딩 모델 초기화 (한 번만 초기화하면 여러 곳에서 재사용 가능)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def truncate_document(document_content):
    """문서 전체 내용을 반환 (토큰 제한 없이)"""
    return document_content

def clean_gpt_output(output):
    """
    GPT 응답에서 코드 블록(백틱으로 감싸진 부분)을 제거하여 순수한 JSON 문자열을 반환합니다.
    예) "```json\n[ ... ]\n```" → "[ ... ]"
    """
    output = output.strip()
    if output.startswith("```"):
        parts = output.split("```")
        if len(parts) >= 3:
            output = parts[1].strip()
    if output.lower().startswith("json"):
        output = output[4:].strip()
    return output

def parse_gpt_response(response_text):
    """GPT 응답 문자열에서 JSON 데이터를 추출하여 파싱합니다."""
    try:
        cleaned_text = clean_gpt_output(response_text)
        if not cleaned_text:
            return {"error": "GPT 응답이 비어 있습니다.", "raw_response": response_text}
        start = cleaned_text.find("[")
        end = cleaned_text.rfind("]") + 1
        if start == -1 or end == -1:
            raise ValueError("GPT 응답에서 JSON 데이터를 찾을 수 없습니다.")
        json_str = cleaned_text[start:end]
        parsed_response = json.loads(json_str)
        if not parsed_response:
            return {"error": "GPT 응답이 빈 JSON 배열입니다.", "raw_response": response_text}
        return parsed_response
    except (json.JSONDecodeError, ValueError) as e:
        return {"error": f"응답을 JSON으로 변환할 수 없습니다: {e}", "raw_response": response_text}

def compute_embedding_score(question, document_text):
    """
    주어진 질문과 문서 전체(또는 핵심 문장)의 임베딩을 생성한 후, 코사인 유사도를 계산합니다.
    반환값은 0~1 사이의 유사도 점수입니다.
    """
    q_emb = embedding_model.encode(question, convert_to_tensor=True)
    doc_emb = embedding_model.encode(document_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(q_emb, doc_emb)
    return score.item()  # tensor 값을 float으로 반환

def analyze_with_gpt(file_type, relevant_docs, document_content):
    """
    GPT를 사용하여 문서 내용을 기반으로 질문에 대한 답변을 생성하고,
    각 질문-답변 쌍에 대해 문서와의 임베딩 유사도(embedding score)도 함께 포함합니다.
    `relevant_docs`는 [{"질문": "질문1"}, {"질문": "질문2"}, ...] 형태여야 합니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
    full_content = truncate_document(document_content)

    if isinstance(relevant_docs, str):
        try:
            relevant_docs = json.loads(relevant_docs)
        except json.JSONDecodeError:
            raise ValueError("relevant_docs 문자열을 JSON으로 변환할 수 없습니다.")

    # 질문 리스트 추출 (입력 사전은 {"질문": "내용"} 형식)
    questions = [{"질문": doc["질문"]} for doc in relevant_docs]

    prompt_template = PromptTemplate(
        input_variables=["file_type", "document_content", "questions"],
        template="""
    주어진 문서 내용을 기반으로 질문에 대한 구체적이고 명확한 답변을 JSON 형식으로 생성하세요.
    
    문서 유형: {file_type}
    문서 내용:
    {document_content}

    질문 목록:
    {questions}

    만약 문서에서 답을 찾을 수 없으면 "답변": "정보 없음"으로 출력하세요.
    
    응답 형식:
    [
        {{
            "질문": "질문 내용",
            "답변": "질문에 대한 구체적이고 명확한 답변"
        }},
        {{
            "질문": "질문 내용",
            "답변": "질문에 대한 구체적이고 명확한 답변"
        }}
    ]
    """
    )

    chain = LLMChain(llm=model, prompt=prompt_template)
    result = chain.run({
        "file_type": file_type,
        "document_content": full_content,
        "questions": json.dumps(questions, ensure_ascii=False)
    })
    
    qa_pairs = parse_gpt_response(result)

    if isinstance(qa_pairs, list):
        for qa in qa_pairs:
            question_text = qa.get("질문", "")
            score = compute_embedding_score(question_text, document_content)
            qa["embedding_score"] = score
    return qa_pairs

def generate_questions(document_content):
    """
    실제 문서 내용을 기반으로 10개의 질문을 생성합니다.
    출력은 JSON 배열, 예: ["질문1", "질문2", ..., "질문10"]
    """
    if not isinstance(document_content, str):
        document_content = str(document_content)
    
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate(
        input_variables=["document"],
        template="""
        다음 문서 내용을 참고하여 10개의 질문을 생성하십시오.

        문서 내용:
        {document}

        질문은 간결하고 의미 있는 형태로 제공되어야 합니다.
        출력 형식 (JSON):
        [
            "질문1",
            "질문2",
            "질문3",
            "질문4",
            "질문5",
            "질문6",
            "질문7",
            "질문8",
            "질문9",
            "질문10"
        ]
        """
    )
    chain = LLMChain(llm=model, prompt=prompt)
    result = chain.run({"document": document_content})
    cleaned_result = clean_gpt_output(result)
    try:
        questions = json.loads(cleaned_result)
        if not isinstance(questions, list) or len(questions) != 10:
            raise ValueError("생성된 질문의 수가 10개가 아닙니다.")
        return questions
    except (json.JSONDecodeError, ValueError) as e:
        return {"error": f"질문 생성 응답을 JSON으로 변환할 수 없습니다: {e}", "raw_response": cleaned_result}

def evaluate_qa_pairs(qa_pairs):
    """
    GPT를 사용하여 각 Q&A 쌍을 0-100 사이로 평가하고, 
    평가 결과를 표 형식(각 평가 항목과 항목별 점수를 포함)으로 제공하며, 
    모든 Q&A 쌍의 평균 점수를 최종 평가 점수로 제시합니다.
    
    평가 기준:
      1. 관련성: 질문과 답변이 문서 내용과 얼마나 관련이 있는지.
      2. 정확성: 답변이 실제 문서의 정보를 정확하게 반영하는지.
      3. 완전성: 답변이 질문에 대해 충분하고 완전한 정보를 제공하는지.
      4. 명료성: 답변이 명확하고 이해하기 쉬운지.
      
    각 Q&A 쌍에 대해 위 네 가지 평가 기준의 개별 점수를 산출한 후, 그 평균을 "score"로 계산해 주세요.
    또한, 모든 Q&A 쌍의 평균 점수를 "overall_score"로 제시해 주세요.
    
    출력 형식은 반드시 아래 예시와 같이 JSON 형식으로 출력해 주세요.
    
    예시 출력:
    {{
         "evaluations": [
             {{
                 "질문": "첫 번째 질문",
                 "답변": "첫 번째 답변",
                 "관련성": 90,
                 "정확성": 85,
                 "완전성": 80,
                 "명료성": 88,
                 "score": 86
             }},
             {{
                 "질문": "두 번째 질문",
                 "답변": "두 번째 답변",
                 "관련성": 80,
                 "정확성": 82,
                 "완전성": 78,
                 "명료성": 85,
                 "score": 81.25
             }}
             // 나머지 Q&A 쌍 평가 결과...
         ],
         "overall_score": 83.5
    }}
    
    입력 데이터:
    {qa_pairs}
    """
    import os
    from openai import OpenAIError
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
         raise OpenAIError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
    prompt = PromptTemplate(
         input_variables=["qa_pairs"],
         template="""
         다음은 문서 기반으로 생성된 10개의 질문과 답변 쌍입니다.
         각 Q&A 쌍에 대해, 아래 평가 기준에 따라 0에서 100 사이의 점수를 부여해 주세요.
         
         평가 기준:
         1. 관련성: 질문과 답변이 문서 내용과 얼마나 관련이 있는지.
         2. 정확성: 답변이 실제 문서의 정보를 정확하게 반영하는지.
         3. 완전성: 답변이 질문에 대해 충분하고 완전한 정보를 제공하는지.
         4. 명료성: 답변이 명확하고 이해하기 쉬운지.
         
         위 네 가지 평가 기준에 대한 개별 점수를 산출한 후, 그 평균을 "score"로 계산해 주세요.
         또한, 모든 Q&A 쌍의 평균 점수를 "overall_score"로 제시해 주세요.
         
         결과는 반드시 아래 예시와 같이 JSON 형식으로 출력해 주세요.
         
         예시 출력:
         {{
             "evaluations": [
                 {{
                     "질문": "첫 번째 질문",
                     "답변": "첫 번째 답변",
                     "관련성": 90,
                     "정확성": 85,
                     "완전성": 80,
                     "명료성": 88,
                     "score": 86
                 }},
                 {{
                     "질문": "두 번째 질문",
                     "답변": "두 번째 답변",
                     "관련성": 80,
                     "정확성": 82,
                     "완전성": 78,
                     "명료성": 85,
                     "score": 81.25
                 }}
                 // 나머지 Q&A 쌍 평가 결과...
             ],
             "overall_score": 83.5
         }}
         
         입력 데이터:
         {qa_pairs}
         """
    )
    chain = LLMChain(llm=model, prompt=prompt)
    result = chain.run({"qa_pairs": json.dumps(qa_pairs, ensure_ascii=False)})
    try:
         evaluation = json.loads(clean_gpt_output(result))
         return evaluation
    except Exception as e:
         return {"error": f"평가 결과를 JSON으로 변환할 수 없습니다: {e}", "raw_response": result}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        document_content = sys.argv[1]
    else:
        document_content = input("문서 내용을 입력하세요: ")

    # 1. 문서 내용을 기반으로 10개의 질문 생성
    questions = generate_questions(document_content)
    print("생성된 질문:", questions)
    
    if isinstance(questions, list) and len(questions) == 10:
        # 2. 생성된 질문을 기반으로 GPT를 통해 답변 생성 (질문-답변 쌍은 단 한 번의 호출로 생성)
        questions_list = [{"질문": q} for q in questions]
        file_type = "PDF"  # 파일 유형은 필요에 따라 지정
        qa_pairs = analyze_with_gpt(file_type, questions_list, document_content)
        print("생성된 Q&A 결과:", qa_pairs)
        # 3. 생성된 Q&A 쌍에 대해 평가 수행
        evaluation = evaluate_qa_pairs(qa_pairs)
        print("평가 결과:", evaluation)
        # 최종 결과를 JSON 파일로 저장 (평가 결과 포함)
        final_result = {
            "qa_pairs": qa_pairs,
            "evaluation": evaluation
        }
        with open("Result.json", "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=4, ensure_ascii=False)
        print("결과가 'Result.json'에 저장되었습니다.")
    else:
        print("질문 생성에 문제가 발생했습니다:", questions)