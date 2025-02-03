from langchain.text_splitter import CharacterTextSplitter

with open("data/appendix-keywords.txt") as f:
  file = f.read()

# CharacterTextSplitter를 사용하여 텍스트를 청크(chunk)로 분할하는 코드
text_splitter = CharacterTextSplitter(
    # 텍스트를 분할할 때 사용할 구분자를 지정합니다. 기본값은 "\n\n" 입니다.
    separator="\n\n",
    # 분할된 텍스트 청크의 최대 크기를 지정합니다, (문자수)
    chunk_size=210,
    # 분할된 텍스트 청크 간의 중복되는 문자 수를 지정합니다.
    chunk_overlap=0,
    # 텍스트의 길이를 계산하는 함수를 지정합니다.
    length_function=len
)


# 텍스트를 청크로 분할합니다.
texts = text_splitter.create_documents([file])
print(len(texts[0].page_content)) # 분할된 문서의 개수를 출력합니다.
print(texts[0]) # 분할된 문서 중 첫 번째 문서를 출력합니다.

# 문서에 대한 메타데이터 리스트를 정의합니다.
metadatas = [
    {"document": 1},
    # {"document": 2},
]

documents = text_splitter.create_documents(
    [
        file,
        # file,
    ], # 분할할 텍스트 데이터를 리스트로 전달합니다.
    metadatas=metadatas # 각 문서에 해당하는 메타데이터를 전달합니다.
)

text_splitter.split_text(file)[0]