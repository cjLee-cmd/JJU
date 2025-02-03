with open("c:/Users/kyh97/OneDrive - 전주대학교/전주대/2024 2학기/비트캠프/12월31일/testresume.md", encoding='utf-8') as f:
  md_text = f.read()

#print(md_text)

from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ('#', 'Header 1'),
    ('##', 'Header 2'),
    ('###', 'Header 3'),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on = headers_to_split_on,
    strip_headers = True
)

md_header_splits = markdown_splitter.split_text(md_text)

for header in md_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n==============================\n")
    
# 결과를 파일에 저장
output_file = "./전주대/2024 2학기/비트캠프/12월31일/split_results.txt"
with open(output_file, "w", encoding="utf-8") as out_file:
    for header in md_header_splits:
        out_file.write(f"{header.page_content}\n")
        out_file.write(f"{header.metadata}\n")
        out_file.write("==============================\n")

print(f"결과물이 '{output_file}'에 저장되었습니다.")