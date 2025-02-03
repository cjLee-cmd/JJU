# FILE_PATH = './1월2일/SPRI_AI_Brief_2023년12월호_F.pdf'

# def show_metadata(docs):
#   if docs:
#     print('[metadata]')
#     print(list(docs[0].metadata.keys()))
#     print('\n[examples]')
#     max_key_length = max(len(k) for k in docs[0].metadata.keys())
#     for k, v in docs[0].metadata.items():
#       print(f'{k:<{max_key_length}} : {v}')

# # !pip install -qU langchain_community
# # !pip install -qU pypdf
# # !pip install -qU pymupdf
# # !pip install -qU PyPDFium2
# # !pip install -qU pdfminer.six

# from langchain_community.document_loaders import PyPDFLoader
# # from langchain_community.document_loaders import PyPDFium2Loader
# # from langchain_community.document_loaders import PDFMinerLoader
# # from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader

# loader = PyPDFLoader(FILE_PATH)

# #pdf loader 초기화
# docs = loader.load()

# print(len(docs))
# print(docs[0])
# print(docs[0].page_content)

# print(show_metadata(docs))



from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, PyPDFium2Loader, PDFMinerLoader, PDFMinerPDFasHTMLLoader

def process_pdf_with_all_loaders(file_path):
    loaders = {
        "PyPDFLoader": PyPDFLoader,
        "PyMuPDFLoader": PyMuPDFLoader,
        "PyPDFium2Loader": PyPDFium2Loader,
        "PDFMinerLoader": PDFMinerLoader,
        "PDFMinerPDFasHTMLLoader": PDFMinerPDFasHTMLLoader,
    }

    def show_metadata(docs):
        if docs:
            print('[metadata]')
            print(list(docs[0].metadata.keys()))
            print('\n[examples]')
            max_key_length = max(len(k) for k in docs[0].metadata.keys())
            for k, v in docs[0].metadata.items():
                print(f'{k:<{max_key_length}} : {v}')

    for loader_name, loader_class in loaders.items():
        print(f"\n======== Processing with {loader_name} ========\n")
        try:
            loader = loader_class(file_path)
            docs = loader.load()
            print(f"Number of documents: {len(docs)}")

            if docs:
                print("First document content:")
                print(docs[0].page_content[:10], "...\n")  # Print first 500 characters of the first document
                show_metadata(docs)
            else:
                print("No documents loaded.")
        except Exception as e:
            print(f"Error processing with {loader_name}: {e}")

# Example usage
FILE_PATH = './1월2일/SPRI_AI_Brief_2023년12월호_F.pdf'
process_pdf_with_all_loaders(FILE_PATH)
