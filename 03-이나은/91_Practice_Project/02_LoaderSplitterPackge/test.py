import os
import shutil
from langchain_community.document_loaders import (
    PyPDFLoader, PyMuPDFLoader, PyPDFium2Loader, PDFMinerLoader, PDFPlumberLoader
)

# 파일 확장자에 따른 처리 함수들
def handle_pdf(file_path):
    try:
        print(f"[INFO] Handling PDF: {file_path}")

        # PyPDFLoader 사용
        try:
            loader_pypdf = PyPDFLoader(file_path, extract_images=True)
            documents_pypdf = loader_pypdf.load()
            print("[INFO] Loaded PDF with PyPDFLoader")
        except Exception as e:
            print(f"[ERROR] PyPDFLoader failed: {e}")

        # PyMuPDFLoader 사용
        try:
            loader_pymupdf = PyMuPDFLoader(file_path)
            documents_pymupdf = loader_pymupdf.load()
            print("[INFO] Loaded PDF with PyMuPDFLoader")
        except Exception as e:
            print(f"[ERROR] PyMuPDFLoader failed: {e}")

        # PyPDFium2Loader 사용
        try:
            loader_pypdfium2 = PyPDFium2Loader(file_path)
            documents_pypdfium2 = loader_pypdfium2.load()
            print("[INFO] Loaded PDF with PyPDFium2Loader")
        except Exception as e:
            print(f"[ERROR] PyPDFium2Loader failed: {e}")

        # PDFMinerLoader 사용
        try:
            loader_pdfminer = PDFMinerLoader(file_path)
            documents_pdfminer = loader_pdfminer.load()
            print("[INFO] Loaded PDF with PDFMinerLoader")
        except Exception as e:
            print(f"[ERROR] PDFMinerLoader failed: {e}")

        # PDFPlumberLoader 사용
        try:
            loader_pdfplumber = PDFPlumberLoader(file_path)
            documents_pdfplumber = loader_pdfplumber.load()
            print("[INFO] Loaded PDF with PDFPlumberLoader")
        except Exception as e:
            print(f"[ERROR] PDFPlumberLoader failed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error while handling PDF: {e}")

# 기타 파일 처리 함수들 (HWP, CSV, Excel 등)
def handle_hwp(file_path):
    print(f"[INFO] Handling HWP: {file_path}")

def handle_csv(file_path):
    print(f"[INFO] Handling CSV: {file_path}")

def handle_excel(file_path):
    print(f"[INFO] Handling Excel: {file_path}")

def handle_txt(file_path):
    print(f"[INFO] Handling TXT: {file_path}")

def handle_json(file_path):
    print(f"[INFO] Handling JSON: {file_path}")

def handle_py(file_path):
    print(f"[INFO] Handling Python: {file_path}")

def handle_html(file_path):
    print(f"[INFO] Handling HTML: {file_path}")

def handle_md(file_path):
    print(f"[INFO] Handling MD: {file_path}")

# 파일 확장자별로 처리할 함수를 매핑
file_handlers = {
    '.pdf': handle_pdf,
    '.hwp': handle_hwp,
    '.csv': handle_csv,
    '.xlsx': handle_excel,
    '.xls': handle_excel,
    '.txt': handle_txt,
    '.json': handle_json,
    '.py': handle_py,
    '.html': handle_html,
    '.md': handle_md,
}

# 파일 정리 및 처리 함수
def sort_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        if os.path.isfile(file_path):
            file_extension = os.path.splitext(filename)[1].lower()

            # 확장자에 맞는 처리 함수 호출
            if file_extension in file_handlers:
                print(f"[INFO] Processing: {filename}")
                try:
                    file_handlers[file_extension](file_path)

                    # 파일 이동
                    destination_path = os.path.join(output_dir, filename)
                    if not os.path.exists(destination_path):
                        shutil.move(file_path, destination_path)
                        print(f"[INFO] Moved {filename} to {output_dir}")
                    else:
                        print(f"[WARNING] File already exists in destination: {filename}")
                except Exception as e:
                    print(f"[ERROR] Failed to process {filename}: {e}")
            else:
                print(f"[WARNING] Unsupported file type: {filename}")
        else:
            print(f"[INFO] Skipping directory: {filename}")

# 사용 예시
input_directory = '/workspaces/JJU-1/03-이나은/Data'
output_directory = '/workspaces/JJU-1/03-이나은/Data/Sorted_Files'

sort_files(input_directory, output_directory)
