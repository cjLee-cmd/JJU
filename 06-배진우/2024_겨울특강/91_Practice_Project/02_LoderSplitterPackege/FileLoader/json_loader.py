"""
JsonLoader.py
- JSON 파일을 직접 열어서 파싱하는 로직.
"""

import json
from pathlib import Path
from pprint import pprint

def load_json(file_path: str):
    """
    JSON 파일을 열어서 파싱 후 파이썬 객체로 반환.
    """

    data = json.loads(Path(file_path).read_text())

    return data


if __name__ == "__main__":
    file_path = "/workspaces/JJU-1/06-배진우/2024_겨울특강/91_Practice_Project/02_LoderSplitterPackege/FileLoder/data/001-BH12(1)-R01-01-90(NA)-D-1-10.json"
    print(load_json(file_path))