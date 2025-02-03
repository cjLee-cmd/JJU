# with open("./html_jju.txt") as f:
#   html_text = f.read()
with open("c:/Users/kyh97/OneDrive - 전주대학교/전주대/2024 2학기/비트캠프/12월30일/html_jju.txt", encoding="utf-8") as f:
  html_text = f.read()

from langchain_text_splitters import(
    Language,
    RecursiveCharacterTextSplitter,
)

#print(html_text)
html_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.HTML,
    chunk_size=60,
    chunk_overlap=0,
)

html_docs = html_splitter.create_documents([html_text])
print(html_docs)