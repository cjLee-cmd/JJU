import os
from dotenv import load_dotenv  # 이 라인을 추가하세요.
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class PDFEvaluator:
    def __init__(self):
        """
        PDF 평가를 위한 클래스 초기화.
        OpenAI API 키는 .env 파일에서 불러옵니다.
        """
        # .env 파일에서 환경 변수 로드
        load_dotenv()  # .env 파일에서 환경 변수를 불러옵니다.
        
        # OpenAI API 키 설정
        self.api_key = os.getenv('OPENAI_API_KEY')  # .env 파일에서 가져온 API 키
        
        if not self.api_key:
            raise ValueError("OpenAI API 키가 .env 파일에 없습니다.")
        
        # OpenAI API 키 설정
        os.environ['OPENAI_API_KEY'] = self.api_key

    def evaluate(self, extracted_texts):
        """
        추출된 텍스트를 평가하여 GPT 모델을 통해 결과를 출력
        :param extracted_texts: 추출된 텍스트
        :return: GPT 모델의 평가 결과
        """
        # 모든 로더의 텍스트를 결합하여 하나의 평가 요청으로 전달
        combined_text = "\n\n".join(extracted_texts)

        # PromptTemplate 정의
        prompt_template_1 = PromptTemplate.from_template(
           "{pdf}를 다음 기준에 따라 성능을 평가하여 표로 정리해 주세요: "
           "1. **정확도 (Accuracy)**: 추출된 텍스트와 원본 PDF 텍스트의 일치 정도 (단어 및 문장 단위 비교) "
           "2. **오류율 (Error Rate)**: 추출된 텍스트에서 발생한 오류의 비율 (예: 잘못 추출된 단어, 문장) "
           "3. **표현력 (Readability)**: 추출된 텍스트의 문법적 완성도와 자연스러움"
            "이 기준을 바탕으로 각 라이브러리의 성능을 비교하여 수치화된 표로만만 작성해 주세요."
        )

        # 모델 설정
        model = ChatOpenAI(
            model='gpt-4o',
            max_tokens=2048,
            temperature=0.1
        )

        # 체인 구성 및 실행
        chain = prompt_template_1 | model
        response = chain.invoke({"pdf": combined_text})
        return response
