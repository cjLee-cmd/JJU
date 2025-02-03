import os
from langchain.document_loaders import PyPDFLoader as PyPDFLoaderClass
from langchain.document_loaders import PyMuPDFLoader as PyMuPDFLoaderClass
from langchain_community.document_loaders import PyPDFium2Loader as PyPDFium2LoaderClass
from langchain_community.document_loaders import PDFMinerLoader as PDFMinerLoaderClass
from langchain_community.document_loaders import PDFPlumberLoader as PDFPlumberLoaderClass
from langchain_teddynote.document_loaders import HWPLoader as HWPLoaderClass
from langchain_community.document_loaders.csv_loader import CSVLoader as CSVLoaderClass
from langchain_community.document_loaders import UnstructuredExcelLoader as UnstructuredExcelLoaderClass
from langchain_community.document_loaders import TextLoader as TextLoaderClass
from langchain_community.document_loaders import JSONLoader as JSONLoaderClass
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup, SoupStrainer
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.datasets import load_files
from dotenv import load_dotenv
import json

# .env 파일 로드
load_dotenv()

# API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USER_AGENT = os.getenv("USER_AGENT")

# API 키 설정
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["USER_AGENT"] = USER_AGENT

input_data = "/workspaces/JJU-1/04-유상민/91_Practice_Project/input데이터 모음/보도자료(240430).hwp"

# 1단계: 입력 데이터 유형 확인하기
def detect_input_type(input_data):
    if input_data.startswith("http://") or input_data.startswith("https://"):
        return "URL"
    elif os.path.isfile(input_data):
        return "FILE"
    else:
        return "Unknown"

# 2단계: 파일 분류하기
def detect_file_type(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    file_types = {
        '.pdf': 'PDF',
        '.csv': 'CSV',
        '.xlsx': 'Excel',
        '.xls': 'Excel',
        '.txt': 'Text',
        '.json': 'JSON',
        '.html': 'HTML',
        '.md': 'Markdown',
        '.hwp': 'HWP',
        '.py': 'Python'
    }

    return file_types.get(file_extension, "Unknown File Type")

# 3단계 loader 매핑
def load_with_webbase(url):
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs=dict(
            parse_only=SoupStrainer(
                "div",
                attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
            )
        ),
    )
    docs = loader.load()
    return docs[0].page_content[:1000]

# Loader 함수 정의
def load_with_pypdf(file_path):
    loader = PyPDFLoaderClass(file_path)
    docs = loader.load()
    return docs[1].page_content[:1000]

def load_with_pymupdf(file_path):
    loader = PyMuPDFLoaderClass(file_path)
    docs = loader.load()
    return docs[1].page_content[:1000]

def load_with_pypdfium(file_path):
    loader = PyPDFium2LoaderClass(file_path)
    docs = loader.load()
    return docs[1].page_content[:1000]

def load_with_pdfminer(file_path):
    loader = PDFMinerLoaderClass(file_path)
    docs = loader.load()
    return docs[1].page_content[:1000]

def load_with_pdfplumber(file_path):
    loader = PDFPlumberLoaderClass(file_path)
    docs = loader.load()
    return docs[1].page_content[:1000]

def load_with_hwp(file_path):
    loader = HWPLoaderClass(file_path)
    docs = loader.load()
    return docs[0].page_content[:1000]

def load_with_csv(file_path):
    loader = CSVLoaderClass(file_path)
    docs = loader.load()
    return docs[0].page_content[:1000]

def load_with_excel(file_path):
    loader = UnstructuredExcelLoaderClass(file_path)
    docs = loader.load()
    return docs[0].page_content[:1000]

def load_with_text(file_path):
    loader = TextLoaderClass(file_path)
    docs = loader.load()
    return docs[0].page_content[:1000]

def load_with_json(file_path):
    # jq_schema를 '.'로 설정하면 전체 JSON 콘텐츠를 로드합니다
    loader = JSONLoaderClass(
        file_path=file_path,
        jq_schema='.',
        text_content=False
    )
    docs = loader.load()
    return docs[0].page_content[:1000]

# Loader 매핑
loader_mapping = {
    "PDF": [load_with_pypdf, load_with_pymupdf, load_with_pypdfium, load_with_pdfminer, load_with_pdfplumber],
    "HWP": [load_with_hwp],
    "CSV": [load_with_csv],
    "Excel": [load_with_excel],
    "Text": [load_with_text],
    "JSON": [load_with_json],
    "URL": [load_with_webbase]
}

# 4단계: TextSplitter 매핑 및 호출
splitter_mapping = {
    "PDF": [
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0),
        CharacterTextSplitter(chunk_size=100, chunk_overlap=0),
        TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    ],
    "HWP": [
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0),
        CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ],
    "CSV": [
        CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ],
    "Excel": [
        CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ],
    "Text": [
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0),
        TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    ],
    "JSON": [
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0),
        TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    ],
    "Markdown": [
        MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")])
    ],
    "URL": [
        HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")])
    ]
}

def split_texts(file_type, loader_results):
    splitters = splitter_mapping.get(file_type)
    if not splitters:
        raise ValueError(f"{file_type}에 적합한 TextSplitter가 없습니다.")

    split_results = []
    
    # Markdown 파일인 경우 직접 파일을 읽고 분할
    if file_type == "Markdown":
        with open(loader_results, 'r', encoding='utf-8') as file:
            markdown_content = file.read()
            
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3")
            ]
        )
        try:
            split_results.append({
                'splitter': 'MarkdownHeaderTextSplitter',
                'result': markdown_splitter.split_text(markdown_content)
            })
            # 헤더가 없는 경우를 위한 백업 splitter
            recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
            split_results.append({
                'splitter': 'RecursiveCharacterTextSplitter',
                'result': recursive_splitter.split_text(markdown_content)
            })
        except Exception as e:
            print(f"Markdown splitting 중 오류 발생: {str(e)}")
    else:
        # 다른 파일 타입들은 기존 방식대로 처리
        for splitter in splitters:
            try:
                if isinstance(loader_results, list):
                    for text in loader_results:
                        split_results.append({
                            'splitter': splitter.__class__.__name__,
                            'result': splitter.split_text(text)
                        })
                else:
                    split_results.append({
                        'splitter': splitter.__class__.__name__,
                        'result': splitter.split_text(loader_results)
                    })
            except Exception as e:
                print(f"Splitter {splitter.__class__.__name__} 처리 중 오류 발생: {str(e)}")
                continue
    
    return split_results

# 5단계: GPT 모델로 분석
def analyze_with_gpt(file_type, loader_results_and_split_results):
    model = ChatOpenAI(
        model="gpt-4o",
        max_tokens=2048,
        temperature=0.7
    )

    loader_results, split_results = loader_results_and_split_results

    # 사용된 loader 정보 추출
    used_loader = "Unknown"
    if file_type in loader_mapping:
        used_loader = loader_mapping[file_type][0].__name__

    # 사용된 splitter 목록 생성
    splitter_list = "\n".join([f"- {result['splitter']}" for result in split_results])

    prompt_template = PromptTemplate(
        input_variables=["file_type", "used_loader", "splitter_list", "split_results"],
        template="""
당신은 파일 분석 전문가입니다. {file_type} 파일에 대한 분석을 수행해주세요.

실제 사용된 Loader:
{used_loader}

사용 가능한 Splitter 목록:
{splitter_list}

분석해야 할 내용:
{split_results}

다음 형식으로 상세한 분석 결과를 제공해주세요:

1. 사용된 Loader 분석:
[실제 사용된 {used_loader}의 성능과 적합성 평가]

2. 가장 적합한 Splitter와 그 이유:
[사용된 Splitter 중 가장 효과적인 것을 선택하고 그 이유를 설명]

3. 데이터의 주요 내용 요약:
[텍스트 내용의 주요 포인트 요약]

4. 데이터 품질 평가 점수:
| 항목 | 점수 (0~100) | 설명 |
|------|--------------|------|
| 텍스트 정확성 | XX | 설명 |
| 세부 정보 보존 | XX | 설명 |
| 구조적 완성도 | XX | 설명 |
| 가독성 | XX | 설명 |
| 문맥 일관성 | XX | 설명 |

5. 데이터 품질에 대한 종합적 평가:
[전반적인 데이터 품질에 대한 평가 및 개선점 제안]
"""
    )

    chain = LLMChain(llm=model, prompt=prompt_template)

    result = chain.run({
        "file_type": file_type,
        "used_loader": used_loader,
        "splitter_list": splitter_list,
        "split_results": str(split_results)[:1000]  # 결과 일부만 전달
    })

    return result

def format_final_output(input_type, file_type, loader_results, splitter_results, gpt_result):
    output = {
        "Input File Type": file_type,
        "Used Loader": None,  # "Selected Loader" 대신 "Used Loader"로 변경
        "Loader Analysis": None,  # 새로운 필드 추가
        "Selected Splitter": None,
        "Selected Splitter Reasoning": None,
        "Analysis Summary": None,
        "Score Table": None,
        "Quality Assessment": None
    }
    
    # GPT 결과를 줄 단위로 분리
    result_lines = gpt_result.split('\n')
    
    current_section = None
    section_content = []
    
    for line in result_lines:
        if line.startswith("1. 사용된 Loader"):  # 섹션 이름 변경
            current_section = "loader"
            continue
        elif line.startswith("2. 가장 적합한 Splitter"):
            if current_section == "loader":
                output["Loader Analysis"] = "\n".join(section_content).strip()
            current_section = "splitter"
            section_content = []
            continue
        elif line.startswith("3. 데이터의 주요 내용"):
            if current_section == "splitter":
                output["Selected Splitter"] = "\n".join(section_content).strip()
            current_section = "summary"
            section_content = []
            continue
        elif line.startswith("4. 데이터 품질 평가"):
            if current_section == "summary":
                output["Analysis Summary"] = "\n".join(section_content).strip()
            current_section = "scores"
            section_content = []
            continue
        elif line.startswith("5. 데이터 품질에 대한"):
            if current_section == "scores":
                output["Score Table"] = "\n".join(section_content).strip()
            current_section = "assessment"
            section_content = []
            continue
        elif line.strip():
            section_content.append(line.strip())
    
    # 마지막 섹션 처리
    if file_type in loader_mapping:
            output["Used Loader"] = loader_mapping[file_type][0].__name__
    
    return output

def print_formatted_output(output):
    print("\n=== Final Analysis Results ===")
    print(f"\nInput File Type: {output['Input File Type']}")
    
    print("\nUsed Loader:")
    print(output['Used Loader'])
    
    print("\nLoader Analysis:")
    print(output['Loader Analysis'])
    
    print("\nSelected Splitter:")
    print(output['Selected Splitter'])
    
    print("\nAnalysis Summary:")
    print(output['Analysis Summary'])
    
    print("\nScore Table:")
    print(output['Score Table'])
    
    print("\nQuality Assessment:")
    print(output['Quality Assessment'])
# 메인 실행 부분도 수정
if __name__ == "__main__":
    input_type = detect_input_type(input_data)
    file_type = detect_file_type(input_data) if input_type == "FILE" else "URL"
    
    print(f"1. Input 데이터 유형: {input_type}")
    print(f"2. 파일 타입: {file_type}")
    
    try:
        # Markdown 파일인 경우 loader 단계 건너뛰기
        if file_type == "Markdown":
            loader_results = input_data  # 파일 경로를 그대로 전달
            print("3. Markdown 파일: Loader 단계 생략")
        else:
            # 기존 loader 로직
            if input_type == "FILE":
                loader_results = loader_mapping[file_type][0](input_data)
                print(f"3. Loader 적용 완료: {loader_mapping[file_type][0].__name__}")
            elif input_type == "URL":
                loader_results = load_with_webbase(input_data)
                print("3. WebBase Loader 적용 완료")
            else:
                raise ValueError("지원하지 않는 입력 유형입니다.")
        
        # Splitter 적용
        print("4. Splitter 적용 중...")
        split_results = split_texts(file_type, loader_results)
        print(f"- {len(split_results)}개의 Splitter 적용 완료")
        
        # GPT 분석
        print("5. GPT 분석 중...")
        gpt_result = analyze_with_gpt(file_type, (loader_results, split_results))
        
        # 최종 결과 형식화
        final_output = format_final_output(input_type, file_type, loader_results, split_results, gpt_result)
        
        # 결과 출력
        print_formatted_output(final_output)

        print("\n=== 분석 완료 ===")
        print("모든 단계가 성공적으로 완료되었습니다.")

    except Exception as e:
        print(f"\nError 발생: {e}")
        print("실행 중 오류가 발생했습니다. 입력 데이터와 환경 설정을 확인해주세요.")
        
        # 상세한 에러 정보 출력
        import traceback
        print("\n상세 에러 정보:")
        print(traceback.format_exc())