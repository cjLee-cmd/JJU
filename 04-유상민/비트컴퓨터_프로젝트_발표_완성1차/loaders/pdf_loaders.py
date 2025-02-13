from langchain_community.document_loaders import PyPDFLoader as PyPDFLoaderClass
from langchain_community.document_loaders import PyMuPDFLoader as PyMuPDFLoaderClass
from langchain_community.document_loaders import PyPDFium2Loader as PyPDFium2LoaderClass
from langchain_community.document_loaders import PDFMinerLoader as PDFMinerLoaderClass
from langchain_community.document_loaders import PDFPlumberLoader as PDFPlumberLoaderClass

def load_file(file_type, file_path):
    loader_mapping = {
        "PDF": load_with_pypdf,
        "MuPDF": load_with_pymupdf,
        "PDFium": load_with_pypdfium,
        "PDFMiner": load_with_pdfminer,
        "PDFPlumber": load_with_pdfplumber
    }
    if file_type not in loader_mapping:
        raise ValueError(f"Unsupported file type: {file_type}")

    loader_function = loader_mapping[file_type]
    docs_content = loader_function(file_path)
    
    # 디버깅용 출력
    print(f"Raw content from loader ({file_type}): {docs_content[:500]}")  # 최대 500자 출력

    if not docs_content.strip():
        raise ValueError(f"Loader failed to extract content from the file: {file_path}")

    return docs_content


def load_with_pypdf(file_path):
    loader = PyPDFLoaderClass(file_path)
    docs = loader.load()
    if not docs:
        print(f"PyPDFLoader: No documents found in {file_path}")
    else:
        for idx, doc in enumerate(docs[:3]):
            print(f"Document {idx}: {doc.page_content[:10000]}")
    return "\n".join(doc.page_content[:10000] for doc in docs[:10] if doc.page_content)


def load_with_pymupdf(file_path):
    loader = PyMuPDFLoaderClass(file_path)
    docs = loader.load()
    return "\n".join(doc.page_content[:10000] for doc in docs[:10] if doc.page_content)

def load_with_pypdfium(file_path):
    loader = PyPDFium2LoaderClass(file_path)
    docs = loader.load()
    return "\n".join(doc.page_content[:10000] for doc in docs[:10] if doc.page_content)

def load_with_pdfminer(file_path):
    loader = PDFMinerLoaderClass(file_path)
    docs = loader.load()
    return "\n".join(doc.page_content[:10000] for doc in docs[:10] if doc.page_content)

def load_with_pdfplumber(file_path):
    loader = PDFPlumberLoaderClass(file_path)
    docs = loader.load()
    return "\n".join(doc.page_content[:10000] for doc in docs[:10] if doc.page_content)

