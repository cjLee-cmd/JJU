# evaluate.py

import json
import re
from config import OPENAI_API_KEY, MODEL_NAME
from langchain_community.chat_models import ChatOpenAI

def evaluate_loader_splitter(content: str) -> dict:
    """
    GPT를 사용하여 loader와 splitter의 조합 결과물을 평가합니다.
    각 항목별로 구체적인 기준에 따라 점수를 부여하고, 
    최종 점수는 각 항목의 가중치 평균으로 계산합니다.
    예시 응답:
    {
        "final_score": 85,
        "details": {
            "structure": 90,
            "context": 80,
            "chunk_size": 70,
            "continuity": 85,
            "readability": 95
        }
    }
    """
    evaluation_prompt = f"""
다음 텍스트에 대해 아래 평가 기준에 따라 세부적으로 점수를 부여해 주세요.

1. **문서 구조 및 포맷 보존**  
   - 원본 문서의 제목, 단락, 리스트 등 구조가 잘 유지되었으면 90~100점, 약간 손상되었으면 70~80점, 전혀 보존되지 않으면 30~40점을 부여합니다.

2. **문맥 및 내용 완성도**  
   - 텍스트가 문서의 주요 내용을 누락 없이 전달하고, 논리적 흐름이 있으면 90~100점, 일부 누락이나 흐름이 어색하면 60~70점, 주요 내용이 빠졌으면 30~40점을 부여합니다.

3. **청크 크기 적절성**  
   - 청크가 정보 전달에 적합한 길이로 나뉘었으면 80~100점, 너무 작거나 크면 40~60점 범위로 평가합니다.

4. **청크 간 연속성 및 정보 전달 일관성**  
   - 청크들이 서로 자연스럽게 이어지면 80~100점, 단절되거나 연결이 어색하면 40~60점을 부여합니다.

5. **텍스트 가독성 및 명료성**  
   - 문법, 표현이 명확하고 읽기 쉬우면 90~100점, 약간 혼란스럽거나 어색하면 60~70점, 이해하기 어렵다면 30~40점을 부여합니다.

각 항목별로 위와 같이 구체적으로 평가해 주세요.  
최종 점수는 (문서 구조 20%, 문맥 완성도 30%, 청크 크기 15%, 청크 연속성 15%, 가독성 20%)의 가중치를 적용한 가중 평균으로 계산해 주세요.

아래는 평가 대상 **전체 텍스트**입니다:
{content}

응답은 반드시 아래 JSON 형식을 따르도록 해 주세요:
{{
    "final_score": <최종 점수>,
    "details": {{
        "structure": <문서 구조 점수>,
        "context": <문맥 완성도 점수>,
        "chunk_size": <청크 크기 점수>,
        "continuity": <청크 연속성 점수>,
        "readability": <가독성 점수>
    }}
}}
    """
    chat_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=MODEL_NAME)
    response = chat_llm.predict(evaluation_prompt)
    
    try:
        json_str_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group()
            eval_result = json.loads(json_str)
            if "final_score" in eval_result and "details" in eval_result:
                return eval_result
        return {"final_score": 50, "details": {
            "structure": 50,
            "context": 50,
            "chunk_size": 50,
            "continuity": 50,
            "readability": 50
        }}
    except Exception as e:
        print("평가 결과 파싱 중 에러:", e)
        return {"final_score": 50, "details": {
            "structure": 50,
            "context": 50,
            "chunk_size": 50,
            "continuity": 50,
            "readability": 50
        }}

if __name__ == "__main__":
    sample_text = ("여기에 평가할 텍스트를 입력하세요. 이 텍스트는 loader와 splitter 조합 결과물에 따라 "
                   "GPT가 평가할 텍스트 예시입니다. 각 항목별로 세부 평가 기준에 따라 점수를 부여해 주세요.")
    result = evaluate_loader_splitter(sample_text)
    print("평가 결과:", result)