with open("c:/Users/kyh97/OneDrive - 전주대학교/전주대/2024 2학기/비트캠프/12월27일/appendix-keywords.txt", encoding="utf-8") as f:
  file = f.read()

print(file[:50], '\n')

from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# CharacterTextSplitter를 사용하여 텍스트를 청크로 분할하는 코드
text_splitter_Character = CharacterTextSplitter(
    separator = "\n\n",     # 텍스트를 분할할 때 사용할 구분자를 지정 (default = '\n\n')
    chunk_size = 210,       # 분할된 텍스트 청크의 최대 크기를 지정 (문자 수)
    chunk_overlap  = 0,     # 분할된 텍스트 청크 간의 중복되는 문자 수를 지정
    length_function = len,  # 텍스트의 길이를 계산하는 함수를 지정
)

text_splitter_Recursive = RecursiveCharacterTextSplitter(
    chunk_size = 150,                             # 청크 크기 설정
    chunk_overlap = 0,                            # 청크 간 중복 문자 수 설정
    length_function = len,                        # 문자열 길이 계산 함수 지정
    is_separator_regex= False,                    #구분자로 정규식을 사용할지 여부를 설정
)


texts_Character = text_splitter_Character.create_documents([file])  # 텍스트를 청크로 분할합니다.
texts_Recursive = text_splitter_Recursive.create_documents([file])

print('\n---------------------------------------------------\n')
print(len(texts_Character[0].page_content))               # 분할된 문서의 개수를 출력합니다.
print('\n---------------------------------------------------\n')
print(texts_Character[0])                                 # 분할된 문서 중 첫 번째 문서를 출력합니다.
print('\n---------------------------------------------------\n')
print(len(texts_Recursive[0].page_content))               # 분할된 문서의 개수를 출력합니다.
print('\n---------------------------------------------------\n')
print(texts_Recursive[0])                                 # 분할된 문서 중 첫 번째 문서를 출력합니다.