FILE_PATH = 'data/SPRI_AI_Brief_2023년12월호_F.pdf'

def show_metadata(docs):
    if docs:
        print("[metadata]")
        print(list(docs[0].metadata.keys()))
        print("\n[examples]")
        max_key_length = max(len(k) for k in docs[0].metadata.keys())
        for k, v in docs[0].metadata.items():
            print(f"{k:<{max_key_length}} : {v}")

def pypdfloader(FILE_PATH):
    from langchain_community.document_loaders import PyPDFLoader

    # 파일 경로 설정
    loader = PyPDFLoader(FILE_PATH)

    # PDF 로더 초기화
    docs = loader.load()

    # 문서의 내용 출력
    print(docs[0].page_content[:300])

    # 메타데이터 출력
    show_metadata(docs)

def pymupdfloader(FILE_PATH):
    from langchain_community.document_loaders import PyMuPDFLoader

    loader = PyMuPDFLoader(FILE_PATH)

    docs = loader.load()
    
    # 문서의 내용 출력
    print(docs[0].page_content[:300])

    show_metadata(docs)

def pypdfium2(FILE_PATH):
    from langchain_community.document_loaders import PyPDFium2Loader

    loader = PyPDFium2Loader(FILE_PATH)

    docs = loader.load()

    # 문서의 내용 출력
    print(docs[0].page_content[:300])

    show_metadata(docs)

def pdfMiner(FILE_PATH):
    from langchain_community.document_loaders import PDFMinerLoader

    loader = PDFMinerLoader(FILE_PATH)

    docs = loader.load()

    # 문서의 내용 출력
    print(docs[0].page_content[:300])

    show_metadata(docs)

def pdfminerpdfashtml(FILE_PATH):
    from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader

    loader = PDFMinerPDFasHTMLLoader(FILE_PATH)

    docs = loader.load()

    # 문서의 내용 출력
    print(docs[0].page_content[:300])

    show_metadata(docs)

pypdfloader(FILE_PATH)
pymupdfloader(FILE_PATH)
pypdfium2(FILE_PATH)
pdfMiner(FILE_PATH)
pdfminerpdfashtml(FILE_PATH)