FILE_PATH = './전주대/2024 2학기/비트캠프/1월2일/SPRI_AI_Brief_2023년12월호_F.pdf'

from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader(file_path=FILE_PATH)

# 1.1 load()
docs = loader.load()

len(docs)
docs[0]

# 1.2 TextSplit()
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)

loader = PyPDFLoader(file_path=FILE_PATH)
split_docs = loader.load_and_split(text_splitter=text_splitter)

len(split_docs)
split_docs[0]

# 1.4 lazy_load
loader.lazy_load()

for doc in loader.lazy_load():
  print(doc)
  print('='*20)

# 1.5 aload()
adocs = loader.aload()

#문서 로드
#await adocs  #docs 다 들어올때까지 기다리는게 await