from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
)
from langchain_community.text_splitter import (
    SemanticChunker,
    CodeSplitter,
    RecursiveJsonSplitter,
)

# Splitter 설정
def get_splitter(file_extension):
    splitters = {
        '.pdf': [
            RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200),
            TokenTextSplitter(chunk_size=500),
        ],
        '.hwp': [
            RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150),
        ],
        '.csv': [
            CharacterTextSplitter(chunk_size=500, chunk_overlap=50),
        ],
        '.xlsx': [
            CharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
        ],
        '.txt': [
            RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200),
            SemanticChunker(),
        ],
        '.json': [
            RecursiveJsonSplitter(),
        ],
        '.html': [
            HTMLHeaderTextSplitter(),
        ],
        '.md': [
            MarkdownHeaderTextSplitter(),
        ],
    }
    return splitters.get(file_extension, [])

# 데이터 분할 함수
def split_content(file_extension, content):
    splitters = get_splitter(file_extension)
    if not splitters:
        raise ValueError(f"No suitable splitter found for file type: {file_extension}")

    split_results = {}
    for splitter in splitters:
        splitter_name = type(splitter).__name__
        if isinstance(content, list):  # Content is a list of documents
            split_results[splitter_name] = [splitter.split_text(doc.page_content) for doc in content]
        elif isinstance(content, dict):  # Content is a dict of loaders
            split_results[splitter_name] = {
                key: [splitter.split_text(doc.page_content) for doc in value]
                for key, value in content.items()
            }
        else:  # Content is a single document
            split_results[splitter_name] = splitter.split_text(content)

    return split_results

# 파일 로드 및 분할
def process_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # 로드 단계
    content = load_file(file_path)

    # 분할 단계
    try:
        split_results = split_content(file_extension, content)
        for splitter_name, chunks in split_results.items():
            print(f"\n=== Split by {splitter_name} ===")
            for chunk in chunks:
                print(chunk)
    except ValueError as e:
        print(f"Error during splitting: {e}")

# Example 실행
try:
    process_file(FILE_PATH)
except ValueError as e:
    print(e)
