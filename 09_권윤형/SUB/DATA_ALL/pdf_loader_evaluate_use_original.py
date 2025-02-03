FILE_PATH = r'C:\Users\kyh97\OneDrive\문서\GitHub\test_JJU\pdf_lecture\2024년도_전략기술_테마별_프로젝트(DCP)_제2차_기술수요조사_시행계획_공고.pdf'

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import PDFPlumberLoader

loader_pypdf = PyPDFLoader(FILE_PATH)
loader_pymupdf = PyMuPDFLoader(FILE_PATH)
loader_pypdfium2 = PyPDFium2Loader(FILE_PATH)
loader_pdfminer = PDFMinerLoader(FILE_PATH)
loader_plumber = PDFPlumberLoader(FILE_PATH)

#pdf loader 초기화
docs_pypdf     = loader_pypdf.load()
docs_pymupdf   = loader_pymupdf.load()
docs_pypdfium2 = loader_pypdfium2.load()
docs_pdfminer  = loader_pdfminer.load()
docs_plumber   = loader_plumber.load()

docs_pypdf_500 = docs_pypdf[0].page_content[:500]
docs_pymupdf_500 = docs_pymupdf[0].page_content[:500]
docs_pypdfium2_500 = docs_pypdfium2[0].page_content[:500]
docs_pdfminer_500 = docs_pdfminer[0].page_content[:500]
docs_plumber_500 = docs_plumber[0].page_content[:500]


output_file = "./pdf_lecture/PDF_Loader_result.txt"
with open(output_file, "w", encoding="utf-8") as out_file:
    out_file.write("=== PyPDFLoader ===\n")
    out_file.write(docs_pypdf_500 + "\n\n")
    
    out_file.write("=== PyMuPDFLoader ===\n")
    out_file.write(docs_pymupdf_500 + "\n\n")
    
    out_file.write("=== PyPDFium2Loader ===\n")
    out_file.write(docs_pypdfium2_500 + "\n\n")
    
    out_file.write("=== PDFMinerLoader ===\n")
    out_file.write(docs_pdfminer_500 + "\n\n")
    
    out_file.write("=== pdfplumber ===\n")
    out_file.write(docs_plumber_500 + "\n\n")

#print(f"결과물이 '{output_file}'에 저장되었습니다.")

# with open("./pdf_lecture/PDF_Loader_result.txt") as f:
#   result_load = f.read()
with open("./pdf_lecture/PDF_Loader_result.txt", "r", encoding="utf-8") as f:
    result_load = f.read()

original = '''
중소벤처기업부 공고 제2024–574호
2024년도 전략기술 테마별 프로젝트(DCP*)
제2차 기술수요조사 시행계획 공고
    * DCP : Deep-tech Challenge Project 
2024년도 전략기술 테마별 프로젝트(DCP)제2차 기술수요조사 시행
계획을 다음과 같이 공고하오니,많은 참여와 신청 바랍니다.
 2024년 11월 18일
중소벤처기업부 장관
< 프로젝트 개요 >
 □프로젝트 소개
◦민간의 先투자를 통해 중소벤처기업의 도전적․혁신적 R&D에 초기 대규모 
R&D예산을 지원함으로써 기술적 난제를 해결하고 중소벤처기업의 혁신적 
성장 지원과 생태계 강화
◦본 공고는 도전적 목표와 파급력 있는 프로젝트에 적합한 수요기술을 공모하기 
위한 기술수요조사 공고이며,기술수요조사를 통해 지정공모 과제(RFP)를기획․
확정하고 ’25년에 지원할 예정
□프로젝트 목적
◦기술개발 역량이 우수한 중소벤처기업이 중심이 되어 전략기술 테마별프로젝트를 
추진할 수 있도록 지원
□지원 대상 :중소벤처기업 주관(대학·출연연·전문연등 컨소시엄 참여 필수)
      * 해외 대학 또는 연구기관 등과 글로벌 협력 연구 권장
□지원 분야 : 12대 국가전략기술 및탄소중립 분야 주요 핵심기술
□추진 절차
기술수요조사
공고➡테마기술
발굴 및 검증➡지정공모 프로젝트 
과제기획 및 출제➡사업공고
(성장전략서·사업계획서)
 
기술개발 수행 및 
PM 전주기 관리프로젝트 선정역량·심층평가
(기술성·사업성·기업역량) 성장전략서 및 
사업계획서 접수
- 2 
 1. 기술수요조사 개요
□(조사목적)전략기술 테마별 프로젝트 발굴을 위해 기술수요를 파악
하고,도전적 목표를 갖는 RFP기획에 활용
□(조사기간) 2024. 11. 18(월)~2024. 12. 27(금) 18:00까지
□(사업설명회) 2024. 12. 2(월) 11:00~12:30,팁스타운S1지하1층 팁스홀
□(조사대상)전략기술 테마별 프로젝트에 관심이 있는 중소벤처기업,
대학·연구기관,학회·협회 등에 소속된 자
□(조사분야) 12대 국가전략기술 및 탄소중립 분야
◦12대 국가전략기술:반도체․디스플레이,이차전지,차세대 원자력,
첨단 이동수단,우주항공․해양,첨단바이오,사이버보안,수소,인공
지능,차세대 통신,첨단로봇․제조,양자
◦탄소중립 분야:태양광·풍력,무탄소 신전원,전력저장,전력망,
에너지 통합시스템,제로에너지 건물, CCUS등
□(조사항목)기술명,국가과학기술표준분류,지원기술분야,기술개발의
배경·필요성,참여기관과의 협력방안,국내·외 기술 및 시장 동향,
개발내용 및 최종 개발 목표,기대효과 등
□(지원내용)프로젝트당 100억원*내외
    * 민간투자 20억원 이상, 정부매칭(매칭투자 최대 40억원, 출연방식 R&D 최대 36억원)으로 구성
□(지원기간) 3년이내 개발 가능한 기술
□(향후계획)약 2개월간 상세기획 추진
기술수요조사
▶
상세기획
▶
기업 지원계획 공고
‘24.11.18~12.27 ‘25.2월~3월(예정) ‘25.3월(예정)
    * 향후계획은 진행 상황에 따라 변동 가능
- 3 
 2. 기술제안 및 제출요령
□기술제안 기본방향
12대 국가전략기술 및 탄소중립 분야에 민간의 先투자와 정부의 
대규모R&D지원을 통해중소벤처기업의 글로벌 성장 및 생태계 활성화 
제고 가능한 기술
①동 프로젝트의 사업목적에 부합하는 테마기술 발굴․검증 및 기술개발
과제 운영을 감안한 수요 제안
②파괴적․혁신적 R&D과제기획과 개발된 기술을 활용하여 최종제품 
및 서비스 구현까지 중소벤처기업 주도의 명확한 목표가 존재하도록 
과제기획(RFP)전제로 수요 제안
③국내·외 대학 또는 연구기관과의 협업을 통해 중소벤처기업이 주도하는 
글로벌 기술개발 생태계 형성이 가능한 수요 제안
□제출요령
◦(제출기한) 2024. 12. 27(금), 18:00까지
     * 전산 접수기간 : ‘24. 12. 2.(월) ~ 12. 27(금), 18:00 까지
◦(제출방법)온라인 접수 및 신청·등록
범부처통합연구지원시스템(www.iris.go.kr) 접속 → 회원가입 → 로그인 → 홈페이지 
Quick menu에서 수요조사 클릭 → 정부부처 : 중소벤처기업부 / 전문기관 : 중소기업
기술정보진흥원 선택 후 검색 → 공고 선택 → 내용입력 후 파일첨부 → 저장 및 제출
◦(제출서류)제안자가 홈페이지*에서 제출 서식(붙임1, 2)을 다운로드 
받아 작성하고 홈페이지에 제안내용을 온라인 입력 후 제출서류를
첨부하여 등록
     * 범부처통합연구지원시스템(IRIS, http://www.iris.go.kr)
'''

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#from langchain_teddynote.messages import stream_response #스트리밍 출력
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

template = '''
{original}에서 여러 라이브러리로 로드한 후,
 500개만 출력해서 저장한 txt 파일이 {result_load}인데, 
어떤 라이브러리를 사용했을 때, RAG로 사용하기 처리가 어느정도로 잘 되었는지 
여러 요소로 정량 평가한 숫자를 총점을 포함한 표로 만들어 주고, 이유를 설명해줘
'''

prompt = PromptTemplate(
    template=template,
    input_variables=["original"],
    partial_variables={"result_load":result_load} #dictionary 형태
)

model = ChatOpenAI(model_name="gpt-4o-mini",
                   max_tokens=2048,
                   temperature=0.1,
                   api_key = OPENAI_API_KEY)

chain = prompt | model

# country 변수에 입력된 값이 자동으로 치환되어 수행됨
print(chain.invoke(original).content)
