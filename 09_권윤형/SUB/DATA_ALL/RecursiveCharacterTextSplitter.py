with open("./appendix-keywords.txt") as f:
  file = f.read()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150,                             # 청크 크기 설정
    chunk_overlap = 0,                            # 청크 간 중복 문자 수 설정
    length_function = len,                        # 문자열 길이 계산 함수 지정
    is_separator_regex= False,                    # 구분자로 정규식을 사용할지 여부를 설정
)

texts = text_splitter.create_documents([file])

print(len(texts))
print('\n------------------------\n')
print(texts[0].page_content)
print('\n------------------------\n')
print(texts[1].page_content)