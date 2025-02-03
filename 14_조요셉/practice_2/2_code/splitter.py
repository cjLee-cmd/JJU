"""
Load된 문서를 splitter로 나눔
"""

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, CharacterTextSplitter, Language, MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter, RecursiveJsonSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def charcter_text_splitter(separator="\n\n", chunk_size=100, chunk_overlap=30):
    text_splitter = CharacterTextSplitter(
        # 텍스트를 분할할 때 사용할 구분자를 지정합니다. 기본값은 "\n\n"입니다.
        separator=separator,
        # 분할된 텍스트 청크의 최대 크기를 지정합니다.
        chunk_size=chunk_size,
        # 분할된 텍스트 청크 간의 중복되는 문자 수를 지정합니다.
        chunk_overlap=chunk_overlap,
    )
    return text_splitter

def recursive_character_text_splitter(chunk_size=100, chunk_overlap=30):
    text_splitter = RecursiveCharacterTextSplitter(
        # 분할된 텍스트 청크의 최대 크기를 지정합니다.
        chunk_size=chunk_size,
        # 분할된 텍스트 청크 간의 중복되는 문자 수를 지정합니다.
        chunk_overlap=chunk_overlap,
    )
    return text_splitter

def token_text_splitter(chunk_size=100, chunk_overlap=30):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        # 분할된 텍스트 청크의 최대 크기를 지정합니다.
        chunk_size=chunk_size,
        # 분할된 텍스트 청크 간의 중복되는 문자 수를 지정합니다.
        chunk_overlap=chunk_overlap,
    )
    return text_splitter

def semanticchunker():
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    return text_splitter

def python_splitter(chunk_size=100, chunk_overlap=30):
    python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap
)
    return python_splitter

def markdownheadersplitter(headers_to_split_on = [  # 문서를 분할할 헤더 레벨과 해당 레벨의 이름을 정의합니다.
    (
        "#",
        "Header 1",
    ),  # 헤더 레벨 1은 '#'로 표시되며, 'Header 1'이라는 이름을 가집니다.
    (
        "##",
        "Header 2",
    ),  # 헤더 레벨 2는 '##'로 표시되며, 'Header 2'라는 이름을 가집니다.
    (
        "###",
        "Header 3",
    ),  # 헤더 레벨 3은 '###'로 표시되며, 'Header 3'이라는 이름을 가집니다.
]):
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return text_splitter

def htmlheadersplitter(headers_to_split_on = [
    ("h1", "Header 1"),  # 분할할 헤더 태그와 해당 헤더의 이름을 지정합니다.
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]):
    text_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return text_splitter

def recursivejsonsplitter(max_chunk_size=300):
    # JSON 데이터를 최대 300 크기의 청크로 분할하는 RecursiveJsonSplitter 객체를 생성합니다.
    text_splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk_size)
    return text_splitter

def main(text_splitter, docs):
    texts = text_splitter.create_documents(docs)
    return texts