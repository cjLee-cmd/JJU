# with open("./html_jju.txt") as f:
#   html_text = f.read()
with open("c:/Users/kyh97/OneDrive - 전주대학교/전주대/2024 2학기/비트캠프/12월30일/pythoncode.txt", encoding="utf-8") as f:
  PYTHON_CODE = f.read()

from langchain_text_splitters import(
    Language,
    RecursiveCharacterTextSplitter,
)

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=50,
    chunk_overlap=2,
)

phthon_docs = python_splitter.create_documents([PYTHON_CODE])
print(phthon_docs)