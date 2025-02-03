from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    PyPDFium2Loader,
    PDFMinerLoader,
    PDFPlumberLoader,
)
from langchain_openai.chat_models import ChatOpenAI  # 올바른 임포트 경로
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain  # 최신 경로 사용 (deprecated, but kept for logging purposes)
from langchain.chains.base import Chain  # For invoke method
from pathlib import Path
import os
from dotenv import load_dotenv
import logging

def setup_logging():
    # 로깅 설정
    logging.basicConfig(
        filename='evaluation.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

def read_save_PDF():
    # 로깅 설정
    setup_logging()
    
    # .env 파일 로드
    env_path = Path("C:/Users/koll2/OneDrive/program_A/2024_비트컴퓨터/과제/.env")
    load_dotenv(dotenv_path=env_path)
    
    # 파일 경로 설정
    FILE_PATH = "C:/Users/koll2/OneDrive/program_A/2024_비트컴퓨터/data/2025년_수출지원기반활용사업_참여기업_1차_모집공고-최종.pdf"

    # 로더 인스턴스 생성
    loaders = {
        "PyPDF": PyPDFLoader(FILE_PATH),
        "PyMuPDF": PyMuPDFLoader(FILE_PATH),
        "PyPDFium2": PyPDFium2Loader(FILE_PATH),
        "PDFMiner": PDFMinerLoader(FILE_PATH),
        "PDFPlumber": PDFPlumberLoader(FILE_PATH),
    }

    # 첫 번째 페이지만 로드
    loaded_docs = {}
    for loader_name, loader in loaders.items():
        try:
            docs = loader.load()
            if docs:
                # 첫 번째 페이지만 선택
                first_page_doc = docs[0]
                loaded_docs[loader_name] = [first_page_doc]
                # print(f"{loader_name} 로더로 첫 번째 페이지를 성공적으로 로드했습니다.")
                logging.info(f"{loader_name} 로더로 첫 번째 페이지를 성공적으로 로드했습니다.")
            else:
                # print(f"{loader_name} 로더로 로드된 문서가 없습니다.")
                logging.warning(f"{loader_name} 로더로 로드된 문서가 없습니다.")
        except Exception as e:
            # print(f"{loader_name} 로더로 문서를 로드하는 중 오류가 발생했습니다: {e}")
            logging.error(f"{loader_name} 로더로 문서를 로드하는 중 오류가 발생했습니다: {e}")

    # 첫 번째 페이지 내용을 저장하는 함수 정의
    def save_first_page_to_txt(
        loaded_docs,
        output_file,
        max_chars=None
    ):
        """
        모든 로더의 첫 번째 페이지를 하나의 텍스트 파일로 저장합니다.

        :param loaded_docs: 로더별로 로드된 문서 딕셔너리
        :param output_file: 저장할 텍스트 파일 경로 (Path 객체 권장)
        :param max_chars: 저장할 최대 글자 수 (None일 경우 전체 저장)
        """
        # 출력 파일 경로 객체 생성
        output_path = Path(output_file)
        
        # 출력 디렉토리가 존재하지 않으면 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", encoding="utf-8") as file:
            for loader_name, docs in loaded_docs.items():
                # 로더 이름을 헤더로 작성
                file.write(f"=== {loader_name} ===\n\n")
                logging.info(f"=== {loader_name} ===")
                
                # 첫 번째 페이지의 내용을 작성
                content = docs[0].page_content
                if max_chars is not None:
                    content = content[:max_chars]
                file.write(content)
                file.write("\n\n")  # 페이지 구분을 위해 줄바꿈 추가
                logging.info(f"{loader_name} 첫 번째 페이지 내용을 저장했습니다.")
        
        # print(f"모든 로더의 첫 번째 페이지 내용이 '{output_path}' 파일에 성공적으로 저장되었습니다.")
        logging.info(f"모든 로더의 첫 번째 페이지 내용이 '{output_path}' 파일에 성공적으로 저장되었습니다.")

    # 출력 파일 경로 설정
    # Windows 절대 경로 사용 (Path 객체를 사용하여 플랫폼에 독립적으로 관리)
    output_file = Path("C:/Users/koll2/OneDrive/program_A/2024_비트컴퓨터/과제/result.txt")
    
    # 첫 번째 페이지를 result.txt 파일로 저장 (글자 수 제한 적용)
    save_first_page_to_txt(
        loaded_docs, 
        output_file, 
        max_chars=500  # 예: 최대 500자 저장
    )

def evaluate_rag_processing():
    # 로깅 설정
    setup_logging()

    # .env 파일 로드
    env_path = Path("C:/Users/koll2/OneDrive/program_A/2024_비트컴퓨터/과제/api_key.env")
    load_dotenv(dotenv_path=env_path)

    # OpenAI API 키 설정
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        logging.error("OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    # OpenAI 모델 설정
    model = ChatOpenAI(
        model_name="gpt-4o-mini", 
        max_tokens=2048,
        temperature=0.1,
        openai_api_key=openai_api_key  # LangChain의 ChatOpenAI 클래스에 API 키 전달
    )

    # 프롬프트 템플릿 정의
    template = "{text}\nRAG로 사용하기 처리가 어느정도로 잘 되었는지 정량 평가를 해줘. 그리고 RAG 점수를 각각 알려줘"

    prompt = PromptTemplate(
        template=template,
        input_variables=["text"],
    )

    # LLMChain 대신 prompt | llm 사용 (최신 LangChain 권장)
    # chain = LLMChain(prompt=prompt, llm=model)
    # 최신 LangChain에서는 LLMChain 대신 prompt | llm 패턴 사용
    chain = prompt | model

    # result.txt 파일 경로 설정
    result_file_path = Path("C:/Users/koll2/OneDrive/program_A/2024_비트컴퓨터/과제/result.txt")

    # result.txt 파일 내용 읽기
    try:
        with result_file_path.open("r", encoding="utf-8") as file:
            text_content = file.read()
            logging.info(f"'{result_file_path}' 파일을 성공적으로 읽었습니다.")
    except Exception as e:
        # print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        logging.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # 체인 실행 및 결과 얻기
    try:
        result = chain.invoke({"text": text_content})
        logging.info("모델의 정량 평가를 성공적으로 수행했습니다.")
    except Exception as e:
        # print(f"체인을 실행하는 중 오류가 발생했습니다: {e}")
        logging.error(f"체인을 실행하는 중 오류가 발생했습니다: {e}")
        return

    # 결과를 score.md로 저장
    score_file_path = Path("C:/Users/koll2/OneDrive/program_A/2024_비트컴퓨터/과제/score.md")
    score_file_path.parent.mkdir(parents=True, exist_ok=True)  # 디렉토리가 없으면 생성
    try:
        # result가 AIMessage인 경우 .content 속성 사용
        if hasattr(result, 'content'):
            write_content = result.content
        else:
            write_content = str(result)
        
        with score_file_path.open("w", eqncoding="utf-8") as file:
            file.write(write_content)
        # print(f"모델의 정량 평가 결과가 '{score_file_path}' 파일에 성공적으로 저장되었습니다.")
        logging.info(f"모델의 정량 평가 결과가 '{score_file_path}' 파일에 성공적으로 저장되었습니다.")
    except IOError as e:
        # print(f"파일을 저장하는 중 오류가 발생했습니다: {e}")
        logging.error(f"파일을 저장하는 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    # read_save_PDF()
    evaluate_rag_processing()