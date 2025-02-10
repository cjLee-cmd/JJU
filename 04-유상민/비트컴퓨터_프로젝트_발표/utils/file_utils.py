# utils/file_utils.py
import os

def detect_input_type(input_data):
    if input_data.startswith("http://") or input_data.startswith("https://"):
        return "URL"
    elif os.path.isfile(input_data):
        return "FILE"
    else:
        return "Unknown"

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
    }

    return file_types.get(file_extension, "Unknown File Type")