from langchain_text_splitters import MarkdownHeaderTextSplitter

markdwon_document = '# Foo\n\n ## Bar\n\nHi this is Jim  \nHi this is Joe\n\n ## Baz\n\n Hi this is Molly'

headers_to_split_on = [
    ('#', 'Header 1'),
    ('##', 'Header 2'),
    ('###', 'Header 3'),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on = headers_to_split_on,
    strip_headers = True
)

md_header_splits = markdown_splitter.split_text(markdwon_document)

for header in md_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n==============================\n")
    
