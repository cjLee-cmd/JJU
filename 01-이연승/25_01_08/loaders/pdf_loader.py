import fitz  # PyMuPDF
from PyPDF2 import PdfReader  # PyPDF
from pdfminer.high_level import extract_text  # PDFMiner
import pdfplumber
from pypdfium2 import PdfDocument  # PyPDFium2

# 1. PyPDF 로더
def load_pypdf(file_path):
    print(f"PyPDF 로더: {file_path} 파일을 로드합니다.")
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# 2. PyMuPDF 로더
def load_pymupdf(file_path):
    print(f"PyMuPDF 로더: {file_path} 파일을 로드합니다.")
    document = fitz.open(file_path)
    text = ""
    for page in document:
        text += page.get_text()
    return text

# 3. PyPDFium2 로더
def load_pypdfium2(file_path):
    print(f"PyPDFium2 로더: {file_path} 파일을 로드합니다.")
    document = PdfDocument(file_path)
    text = ""
    for page in document:
        text += page.get_textpage().get_text()
    return text

# 4. PDFMiner 로더
def load_pdfminer(file_path):
    print(f"PDFMiner 로더: {file_path} 파일을 로드합니다.")
    text = extract_text(file_path)
    return text

# 5. pdfplumber 로더
def load_pdfplumber(file_path):
    print(f"pdfplumber 로더: {file_path} 파일을 로드합니다.")
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text
