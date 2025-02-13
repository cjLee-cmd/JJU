from langchain_community.document_loaders.csv_loader import CSVLoader as CSVLoaderClass
from langchain_community.document_loaders import UnstructuredExcelLoader as UnstructuredExcelLoaderClass

def load_with_csv(file_path):
    loader = CSVLoaderClass(file_path)
    docs = loader.load()
    return docs[0].page_content[:1000]

def load_with_excel(file_path):
    loader = UnstructuredExcelLoaderClass(file_path)
    docs = loader.load()
    return docs[0].page_content[:1000]
