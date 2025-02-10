# splitters/text_splitters.py
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter
)

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
            recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
            split_results.append({
                'splitter': 'RecursiveCharacterTextSplitter',
                'result': recursive_splitter.split_text(markdown_content)
            })
        except Exception as e:
            print(f"Markdown splitting 중 오류 발생: {str(e)}")
    else:
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