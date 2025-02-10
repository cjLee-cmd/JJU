from .pdf_loaders import load_with_pypdf, load_with_pymupdf, load_with_pypdfium, load_with_pdfminer, load_with_pdfplumber
from .hwp_loaders import load_with_hwp
from .csv_loaders import load_with_csv, load_with_excel
from .json_loaders import load_with_json
from .text_loaders import load_with_text
from .web_loader import load_with_webbase

loader_mapping = {
    "PDF": [load_with_pypdf, load_with_pymupdf, load_with_pypdfium, load_with_pdfminer, load_with_pdfplumber],
    "HWP": [load_with_hwp],
    "CSV": [load_with_csv],
    "Excel": [load_with_excel],
    "Text": [load_with_text],
    "JSON": [load_with_json],
    "URL": [load_with_webbase]
}

def load_file(file_type, file_path):
    if file_type in loader_mapping:
        return loader_mapping[file_type][0](file_path)
    raise ValueError(f"Unsupported file type: {file_type}")
