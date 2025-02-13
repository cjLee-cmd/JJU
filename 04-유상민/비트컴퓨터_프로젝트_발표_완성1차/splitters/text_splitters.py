# splitters/text_splitters.py
import os
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter
)

# 파일 형식별 TextSplitter 매핑
# Markdown의 경우, 두 가지 분할 방법(헤더 기반, Recursive)을 함께 사용합니다.
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
        MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]),
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ],
    "URL": [
        HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")])
    ]
}

def split_texts(file_type, loader_results):
    """
    Args:
        file_type (str): PDF, HWP, JSON, Text, Markdown 등 파일의 타입.
        loader_results: 텍스트 내용(문자열) 혹은 텍스트 조각의 리스트.
                        Markdown의 경우 파일 경로가 전달될 수도 있습니다.
    
    Returns:
        split_results (list): 각 TextSplitter의 이름과 분할 결과를 담은 리스트.
    """
    splitters = splitter_mapping.get(file_type)
    if not splitters:
        raise ValueError(f"{file_type}에 적합한 TextSplitter가 없습니다.")

    split_results = []
    
    # Markdown의 경우, loader_results가 파일 경로일 수 있으므로 존재 여부를 체크합니다.
    if file_type == "Markdown":
        if isinstance(loader_results, str) and os.path.exists(loader_results):
            try:
                with open(loader_results, 'r', encoding='utf-8') as file:
                    content = file.read()
            except Exception as e:
                print(f"Markdown 파일 읽기 중 오류 발생: {str(e)}")
                return []
        else:
            content = loader_results  # 이미 텍스트 내용이 전달된 경우

        for splitter in splitters:
            try:
                result = splitter.split_text(content)
                split_results.append({
                    'splitter': splitter.__class__.__name__,
                    'result': result
                })
            except Exception as e:
                print(f"Splitter {splitter.__class__.__name__} 처리 중 오류 발생: {str(e)}")
    else:
        # loader_results가 리스트인지 확인하여 처리합니다.
        texts = loader_results if isinstance(loader_results, list) else [loader_results]
        for text in texts:
            for splitter in splitters:
                try:
                    result = splitter.split_text(text)
                    split_results.append({
                        'splitter': splitter.__class__.__name__,
                        'result': result
                    })
                except Exception as e:
                    print(f"Splitter {splitter.__class__.__name__} 처리 중 오류 발생: {str(e)}")
                    continue
    
    return split_results