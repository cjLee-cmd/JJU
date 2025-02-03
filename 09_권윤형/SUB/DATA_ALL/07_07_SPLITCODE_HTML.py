from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter

with open("./전주대/2024 2학기/비트캠프/12월31일/notion_html/pretzels.html", encoding='utf-8') as f:
  html_text = f.read()

#url = 'https://plato.stanford.edu/entries/goedel/'

headers_to_split_on = [
    ('h1', 'Header 1'),
    ('h2', 'Header 2'),
    ('h3', 'Header 3'),
    ('h4', 'Header 4'),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

#html_header_splits = html_splitter.split_text_from_url(url)
html_header_splits = html_splitter.split_text(html_text)

chunk_size = 500
chunk_overlap = 30

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

html_header_chunks = text_splitter.split_documents(html_header_splits)

splits = text_splitter.split_documents(html_header_chunks)

for split in splits:
  print(f"{split.page_content}")
  print(f"{split.metadata}", end= '\n=========================\n')

# 결과를 파일에 저장
output_file = "./전주대/2024 2학기/비트캠프/12월31일/HTMLHEADERTEXTSPLITTEXT_RESULT.txt"
with open(output_file, "w", encoding="utf-8") as out_file:
    for header in splits:
        out_file.write(f"{header.page_content}\n")
        out_file.write(f"{header.metadata}\n")
        out_file.write("==============================\n")

print(f"결과물이 '{output_file}'에 저장되었습니다.")
