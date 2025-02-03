with open(r"C:\Users\kowm6\Desktop\html\Jangho CHOI 16d537b2a7d180b384f7d6a9a8185625.html", "r", encoding="utf-8") as f:
    html = f.read()

from langchain_text_splitters import HTMLHeaderTextSplitter

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]


html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on,
    return_each_element=True,
)


html_header_splits_elements = html_splitter.split_text(html)

for header in html_header_splits_elements:
    print(f"Header: {header.page_content}")
    print(f"Element: {header.metadata}", end="\n=========================\n")
    print()