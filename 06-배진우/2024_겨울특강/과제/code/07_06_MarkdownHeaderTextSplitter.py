from langchain_text_splitters import MarkdownHeaderTextSplitter

with open("C:/Users/koll2/OneDrive/Obsidian/4. Archive/앱 만들기.md", encoding="UTF-8") as f:
    file = f.read()

# 마크다운 형식의 문서를 문자열로 정의합니다.
markdown_document = file
headers_to_split_on = [  # 문서를 분할할 헤더 레벨과 해당 레벨의 이름을 정의합니다.
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
]

# 마크다운 헤더를 기준으로 텍스트를 분할하는 MarkdownHeaderTextSplitter 객체를 생성합니다.
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# markdown_document를 헤더를 기준으로 분할하여 md_header_splits에 저장합니다.
md_header_splits = markdown_splitter.split_text(markdown_document)
# 분할된 결과를 출력합니다.
for header in md_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")