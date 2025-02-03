import os

# 원래 에이피아이 키가 있어야하는 자리리
FILE_PATH = 'C:\\Users\\SANGMIN\\OneDrive\\문서\\GitHub\\test_jju\\data\\소상공인_전기요금_특별지원_사업_시행_수정공고(제2024-557호).pdf'

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader, PDFMinerLoader
from langchain_community.document_loaders import PyPDFium2Loader

def extract_text_from_pdf(file_path: str):
    """PDF에서 텍스트 추출"""
    loaders = {
        "PyPDFLoader": PyPDFLoader,
        "PyMuPDFLoader": PyMuPDFLoader,
        "PyPDFium2Loader": PyPDFium2Loader,
        "PDFMinerLoader": PDFMinerLoader
    }

    extracted_texts = []
    for loader_name, loader_class in loaders.items():
        try:
            loader_instance = loader_class(file_path)
            docs = loader_instance.load()
            text = docs[0].page_content[:500]  # 500자 추출
            extracted_texts.append({"모델명": loader_name, "텍스트": text})
        except Exception as e:
            extracted_texts.append({"모델명": loader_name, "텍스트": f"오류 발생: {str(e)}"})

    return extracted_texts


def analyze_with_gpt(texts):
    """GPT를 사용해 정량 평가표 및 피드백 생성"""
    model = ChatOpenAI(
        model="gpt-4o",
        max_tokens=2048,
        temperature=0.1,
    )

    prompt_template = PromptTemplate(
        input_variables=["texts"],
        template=(
            "다음은 여러 PDF 추출 모델에서 가져온 텍스트들입니다:\n\n{texts}\n\n"
            "각 텍스트에 대해 다음을 수행하세요:\n"
            "1. 정량 평가표를 작성 (항목: 텍스트 길이, 문장 구조, 단어 다양성, 특수문자 적절성, 가독성, 총점). "
            "표는 아래 형식으로 작성하세요:\n"
            "| 모델명          | 텍스트 길이 | 문장 구조 | 단어 다양성 | 특수문자 적절성 | 가독성 | 총점 |\n"
            "|-----------------|-------------|-----------|-------------|-----------------|--------|------|\n"
            "\n"
            "2. 각 모델별 피드백을 작성. 피드백은 별도의 목록으로 작성하세요.\n"
            "결과는 Markdown 형식으로 반환하세요."
        )
    )

    chain = LLMChain(llm=model, prompt=prompt_template)

    result = chain.run({"texts": texts})
    return result


if __name__ == "__main__":
    # PDF 파일 경로
    file_path = FILE_PATH

    # PDF 텍스트 추출
    extracted_texts = extract_text_from_pdf(file_path)

    # 모델별 텍스트 출력
    print("\n=== 각 모델별 추출된 텍스트 ===")
    for item in extracted_texts:
        print(f"\n[{item['모델명']}]")
        print(item['텍스트'])
        print('-' * 80)

    # GPT 입력 준비
    gpt_input = "\n".join(
        f"모델명: {item['모델명']}\n텍스트: {item['텍스트']}"
        for item in extracted_texts
    )

    # GPT 분석 실행
    gpt_result = analyze_with_gpt(gpt_input)

    # 결과 출력
    print("\n=== GPT 분석 결과 ===")
    print(gpt_result)
