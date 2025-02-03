# FILE_PATH = './전주대/2024 2학기/비트캠프/1월2일/SPRI_AI_Brief_2023년12월호_F.pdf'

# from langchain_community.document_loaders import PyPDFLoader


# loader = PyPDFLoader(file_path=FILE_PATH)

# # 1.1 load()
# docs = loader.load()

# len(docs)
# docs[0]

# # 1.2 TextSplit()
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=200,
#     chunk_overlap=0,
# )

# loader = PyPDFLoader(file_path=FILE_PATH)
# split_docs = loader.load_and_split(text_splitter=text_splitter)

# len(split_docs)
# split_docs[0]

# # 1.4 lazy_load
# loader.lazy_load()

# for doc in loader.lazy_load():
#   print(doc)
#   print('='*20)

# # 1.5 aload()
# adocs = loader.aload()

# #문서 로드
# await adocs  #docs 다 들어올때까지 기다리는게 await

FILE_PATH = './전주대/2024 2학기/비트캠프/1월2일/SPRI_AI_Brief_2023년12월호_F.pdf'

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio  # 비동기 실행을 위해 asyncio 가져오기

# 비동기 함수 정의
async def main():
    # 1. 파일 로더 초기화
    loader = PyPDFLoader(file_path=FILE_PATH)

    # 2. 문서 로드 (동기 방식)
    docs = loader.load()
    print(f"문서 개수: {len(docs)}")
    print(f"첫 번째 문서: {docs[0]}")

    # 3. 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
    )
    split_docs = loader.load_and_split(text_splitter=text_splitter)
    print(f"분할된 문서 개수: {len(split_docs)}")
    print(f"첫 번째 분할 문서: {split_docs[0]}")

    # 4. Lazy Load (지연 로드)
    for doc in loader.lazy_load():
        print(doc)
        print('=' * 20)

    # 5. 비동기 로드
    adocs = await loader.aload()  # 비동기 로드 대기
    print("비동기로 로드된 문서:")
    print(adocs)

# 스크립트 실행 진입점
if __name__ == "__main__":
    asyncio.run(main())  # 비동기 함수 실행
