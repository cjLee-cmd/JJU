import os
from loader import (
    hwp_loader, csv_loader, unstructured_csv_loader, excel_loader, word_loader,
    unstructured_word_loader, txt_loader, json_loader, python_loader,
    html_loader, markdown_loader
)
from pdf_loader import (
    py_pdf_loader, py_mu_pdf, py_pdf_mu2loader, pdf_minerloader, pdf_plumber
)
from splitter import (
    charcter_text_splitter, recursive_character_text_splitter, token_text_splitter,
    semanticchunker, python_splitter, markdownheadersplitter, htmlheadersplitter,
    recursivejsonsplitter
)

def load_and_split(file_path):
    """
    Loads a file based on its extension and splits it using appropriate loaders and splitters.
    """
    print(f"[디버그] 처리할 파일 경로: {file_path}")
    extension = os.path.splitext(file_path)[-1].lower()
    print(f"[디버그] 파일 확장자: {extension}")

    # Dictionary mapping extensions to loader functions
    loader_map = {
        '.hwp': [hwp_loader],
        '.csv': [csv_loader, unstructured_csv_loader],
        '.xlsx': [excel_loader],
        '.xls': [excel_loader],
        '.docx': [word_loader, unstructured_word_loader],
        '.txt': [txt_loader],
        '.json': [json_loader],
        '.py': [python_loader],
        '.html': [html_loader],
        '.md': [markdown_loader],
        '.pdf': [py_pdf_loader, py_mu_pdf, py_pdf_mu2loader, pdf_minerloader, pdf_plumber]
    }

    # Dictionary mapping extensions to multiple splitter functions
    splitter_map = {
        '.txt': [charcter_text_splitter, token_text_splitter, recursive_character_text_splitter, semanticchunker],
        '.docx': [charcter_text_splitter, token_text_splitter, recursive_character_text_splitter, semanticchunker],
        '.csv': [charcter_text_splitter, token_text_splitter, recursive_character_text_splitter, semanticchunker],
        '.hwp': [charcter_text_splitter, token_text_splitter, recursive_character_text_splitter, semanticchunker],
        '.json': [recursivejsonsplitter, token_text_splitter, recursive_character_text_splitter, semanticchunker],
        '.py': [python_splitter, token_text_splitter, recursive_character_text_splitter, semanticchunker],
        '.md': [markdownheadersplitter, token_text_splitter, recursive_character_text_splitter, semanticchunker],
        '.html': [htmlheadersplitter, token_text_splitter, recursive_character_text_splitter, semanticchunker],
        '.pdf': [charcter_text_splitter, token_text_splitter, recursive_character_text_splitter, semanticchunker]
    }

    if extension not in loader_map:
        raise ValueError(f"Unsupported file extension: {extension}")

    print("[디버그] 로더 시작")
    # Load the document using the appropriate loaders
    loaders = loader_map[extension]
    loader_results = []
    loader_error = []

    for loader in loaders:
        print(f"[디버그] 로더 실행: {loader.__name__}")
        try:
            docs = loader(file_path)
            print(f"[디버그] 로더 성공: {loader.__name__}, 문서 길이: {len(docs)}")
            loader_results.append((loader.__name__, docs))
        except Exception as e:
            print(f"[디버그] 로더 오류: {loader.__name__}, 오류 메시지: {str(e)}")
            loader_error.append((loader.__name__, f"Error: {str(e)}"))

    print("[디버그] 로더 완료, 결과 문서 수: {}".format(len(loader_results)))

    # Split the document using the appropriate splitters
    if extension in splitter_map:
        splitters = splitter_map[extension]
        split_error = {}
        results = {}
        i = 0

        print("[디버그] 스플리터 시작")
        for splitter in splitters:
            print(f"[디버그] 스플리터 실행: {splitter.__name__}")
            try:
                text_splitter = splitter()
                split_docs = [text_splitter.create_documents([doc.page_content]) for name, docs in loader_results if isinstance(docs, list) for doc in docs]
                print(f"[디버그] 스플리터 성공: {splitter.__name__}, 분할된 문서 수: {len(split_docs)}")
                for name, doc in loader_results:
                    if isinstance(doc, list):
                        for do in doc:
                            split_docs = text_splitter.create_documents([do.page_content])
                            result = {"loader" : name, "load_docs" : doc, "splitter" : splitter.__name__, "split_docs" : split_docs}
                            results[i] = result
                            i += 1

            except Exception as e:
                print(f"[디버그] 스플리터 오류: {splitter.__name__}, 오류 메시지: {str(e)}")
                split_error[splitter.__name__] = f"Error: {str(e)}"

        print("[디버그] 스플리터 완료")
        return results

# Example usage
if __name__ == "__main__":
    file_path = "14_조요셉/practice_2/1_usedData/240327_공고_아이비케이벤처투자의 여신전문금융업 등록.pdf"  # Replace with actual file path
    try:
        results = load_and_split(file_path)

        print("result:")
        for number, result in results.items():
            print(f'Loader : {result['loader']}')
            for load_doc in result['load_docs']:
                print(load_doc.page_content)
                print('===' * 20)
                print(f'splitter : {result['splitter']}')
                for split_doc in result['split_docs']:
                    print(split_doc.page_content)
                    print('===' * 20)

    except ValueError as e:
        print("Error:", e)