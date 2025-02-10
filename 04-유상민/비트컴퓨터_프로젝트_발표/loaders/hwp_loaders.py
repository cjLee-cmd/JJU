from langchain_teddynote.document_loaders import HWPLoader as HWPLoaderClass

def load_with_hwp(file_path):
    loader = HWPLoaderClass(file_path)
    docs = loader.load()
    return docs[0].page_content[:1000]
