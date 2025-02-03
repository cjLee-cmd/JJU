# pdf_loader.py
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, PyPDFium2Loader, PDFMinerLoader, PDFPlumberLoader

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

    # 첫 페이지 내용의 500글자만 추출
    docs_pypdf_500 = docs_pypdf[0].page_content[:500]
    docs_pymupdf_500 = docs_pymupdf[0].page_content[:500]
    docs_pypdfium2_500 = docs_pypdfium2[0].page_content[:500]
    docs_pdfminer_500 = docs_pdfminer[0].page_content[:500]
    docs_plumber_500 = docs_plumber[0].page_content[:500]

    # 결과를 딕셔너리로 반환
    return {
        'PyPDFLoader': docs_pypdf_500,
        'PyMuPDFLoader': docs_pymupdf_500,
        'PyPDFium2Loader': docs_pypdfium2_500,
        'PDFMinerLoader': docs_pdfminer_500,
        'PDFPlumber': docs_plumber_500
    }

def save_pdf_results(output_file, result_data):
    with open(output_file, "w", encoding="utf-8") as out_file:
        for loader_name, content in result_data.items():
            out_file.write(f"=== {loader_name} ===\n")
            out_file.write(content + "\n\n")
