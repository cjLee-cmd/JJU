from langchain_community.document_loaders import DirectoryLoader

# 경로를 raw string(r)으로 지정
loader = DirectoryLoader(
    r"C:\Users\SANGMIN\OneDrive\문서\GitHub\test_jju",
    glob="**/*.md"
)

docs = loader.load()

print(len(docs))
print(docs[0].page_content[:100])
