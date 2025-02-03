import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, PyPDFium2Loader, PDFMinerLoader, PDFPlumberLoader
from langchain_community.document_loaders import CSVLoader
from langchain_teddynote.document_loaders import HWPLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#FILE_PATH = '/workspaces/JJU/09_권윤형/practice_2/1_usedData/광화사.txt'
FILE_PATH = '/workspaces/JJU/09_권윤형/practice_2/1_usedData/금융감독원_공지사항.pdf'

def load_pdf(FILE_PATH):
    # 각 로더 초기화
    loader_pypdf = PyPDFLoader(FILE_PATH)
    loader_pymupdf = PyMuPDFLoader(FILE_PATH)
    loader_pypdfium2 = PyPDFium2Loader(FILE_PATH)
    loader_pdfminer = PDFMinerLoader(FILE_PATH)
    loader_plumber = PDFPlumberLoader(FILE_PATH)

    # 각 로더로 PDF 파일을 로드
    docs_pypdf = loader_pypdf.load()
    docs_pymupdf = loader_pymupdf.load()
    docs_pypdfium2 = loader_pypdfium2.load()
    docs_pdfminer = loader_pdfminer.load()
    docs_plumber = loader_plumber.load()

    # 결과를 딕셔너리로 반환
    return {
        'PyPDFLoader': docs_pypdf,
        'PyMuPDFLoader': docs_pymupdf,
        'PyPDFium2Loader': docs_pypdfium2,
        'PDFMinerLoader': docs_pdfminer,
        'PDFPlumber': docs_plumber
    }

def load_hwp(FILE_PATH):
    loader_hwp = HWPLoader(FILE_PATH)
    docs_hwp = loader_hwp.load()
    return docs_hwp

def load_csv(FILE_PATH):
    loader_csv = CSVLoader(FILE_PATH)
    docs_csv = loader_csv.load()
    return docs_csv

def load_excel(FILE_PATH):
    loader_excel = UnstructuredExcelLoader(FILE_PATH)
    docs_excel = loader_excel.load()
    return docs_excel

def load_txt(FILE_PATH):
    loader_txt = TextLoader(FILE_PATH)
    docs_txt = loader_txt.load()
    return docs_txt


def load_json(FILE_PATH):
    loader_json = JSONLoader(FILE_PATH)
    docs_json = loader_json.load()
    return docs_json

def load_html(FILE_PATH):
    loader_html = UnstructuredHTMLLoader(FILE_PATH)
    loader_bshtml = BSHTMLLoader(FILE_PATH)

    docs_html = loader_html.load()
    docs_bshtml = loader_bshtml.load()

    return {
        'HTMLLoader': docs_html,
        'BSHTMLLoader': docs_bshtml,
    }

def load_md(FILE_PATH):
    loader_md = UnstructuredMarkdownLoader(FILE_PATH)
    docs_md = loader_md.load()
    return docs_md

def split_pdf(content):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 150,                             # 청크 크기 설정
        chunk_overlap = 0,                            # 청크 간 중복 문자 수 설정
        length_function = len,                        # 문자열 길이 계산 함수 지정
        is_separator_regex= False,                    #구분자로 정규식을 사용할지 여부를 설정
    )

    texts_recursive = text_splitter.create_documents([content])

    return texts_recursive



def load_file(FILE_PATH):
    _, file_extension = os.path.splitext(FILE_PATH)
    file_extension = file_extension.lower()

    # Mapping file extensions to loader functions
    loaders = {
        '.pdf': load_pdf,
        '.hwp': load_hwp,
        '.csv': load_csv,
        '.xlsx': load_excel,
        '.txt': load_txt,
        '.json': load_json,
        '.md': load_md,
        '.html': load_html,
    }

    if file_extension in loaders:
        return loaders[file_extension](FILE_PATH)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def process_file(FILE_PATH, content):
    _, file_extension = os.path.splitext(FILE_PATH)
    file_extension = file_extension.lower()

    # Mapping file extensions to loader functions
    splitters = {
        '.pdf': split_pdf,
        # '.hwp': split_hwp,
        # '.csv': split_csv,
        # '.xlsx': split_excel,
        # '.txt': split_txt,
        # '.json': split_json,
        # '.md': split_md,
        # '.html': split_html,
    }

    if file_extension in splitters:
        return splitters[file_extension](content)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    # # 분할 단계
    # try:
    #     split_results = split_content(file_extension, content)
    #     for splitter_name, chunks in split_results.items():
    #         print(f"\n=== Split by {splitter_name} ===")
    #         for chunk in chunks:
    #             print(chunk)
    # except ValueError as e:
    #     print(f"Error during splitting: {e}")


# Example usage
#file_path = "example.pdf"  # Replace with your file path

try:
    content = load_file(FILE_PATH)
    print(content)
    #split_content = process_file(FILE_PATH, content)
    #print(split_content)
except ValueError as e:
    print(e)