from langchain_teddynote.document_loaders import HWPLoader

loader = HWPLoader(r'C:\Users\kyh97\OneDrive\문서\GitHub\test_JJU\디지털 정부혁신 추진계획.hwp')

docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)
print(docs[0])