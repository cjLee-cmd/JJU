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
from langchain_text_splitters import CharacterTextSplitter

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
    chunk_size = 150
    chunk_overlap = 0

    # 사용할 splitter 리스트 정의
    splitters = {
        'RecursiveCharacterTextSplitter': RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        'CharacterTextSplitter': CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        'TokenTextSplitter' : CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
    }

    # 각 Splitter 결과를 저장할 딕셔너리
    all_splits = {}

    # PDF 로드 결과가 dict로 반환되므로 필요한 데이터를 추출
    for loader_name, documents in content.items():
        # documents는 문서 리스트이며, 각 문서의 `page_content`를 추출
        page_texts = [doc.page_content for doc in documents]

        # 각 Splitter를 사용해 문서 분할
        loader_results = {}
        for splitter_name, splitter in splitters.items():
            split_result = splitter.create_documents(page_texts)
            loader_results[splitter_name] = split_result

        # 로더 이름별로 분할 결과 저장
        all_splits[loader_name] = loader_results

    return all_splits



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

try:
    content = load_file(FILE_PATH)  # Load the file
    split_content = process_file(FILE_PATH, content)  # Process the file

    # Open a file to write results
    with open("/workspaces/JJU/09_권윤형/practice_2/2_code/result.txt", "w", encoding="utf-8") as result_file:
        for loader_name, splitter_results in split_content.items():
            #result_file.write(f"\n=== Split results for {loader_name} ===\n")
            for splitter_name, chunks in splitter_results.items():
                result_file.write(f"\n=====[Results from {loader_name}, {splitter_name}]=====\n")
                for chunk in chunks:
                    if isinstance(chunk, str):
                        result_file.write(chunk + "\n")  # Write the chunk if it's a string
                    else:
                        result_file.write(chunk.page_content + "\n")  # If it's an object
                    result_file.write("\n-------------------------------------\n")
    
    print("Results saved to result.txt")
except ValueError as e:
    print(e)
