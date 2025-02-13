from langchain_community.document_loaders import TextLoader as TextLoaderClass

def load_with_text(file_path):
    loader = TextLoaderClass(file_path)
    docs = loader.load()
    return docs[0].page_content[:1000]
