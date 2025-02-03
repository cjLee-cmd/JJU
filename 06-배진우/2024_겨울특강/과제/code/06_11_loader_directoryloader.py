import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader

def get_current_script_name():
    """
    현재 실행 중인 스크립트의 파일 이름을 반환합니다.
    확장자를 제외하고 반환합니다.
    """
    return os.path.splitext(os.path.basename(__file__))[0]

def load_text_documents(path='./', extensions=['txt', 'md', 'ipynb'], encoding='utf-8'):
    """
    지정된 경로에서 특정 확장자의 텍스트 파일을 로드합니다.
    
    :param path: 파일을 로드할 디렉토리 경로
    :param extensions: 로드할 파일의 확장자 리스트
    :param encoding: 파일 인코딩
    :return: 로드된 문서 리스트
    """
    data = []
    for ext in extensions:
        glob_pattern = f'**/*.{ext}'
        loader = DirectoryLoader(
            path=path,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={'encoding': encoding}
        )
        try:
            loaded = loader.load()
            print(f"Loaded {len(loaded)} documents with extension .{ext}")
            data.extend(loaded)
        except Exception as e:
            print(f"Error loading files with pattern {glob_pattern}: {e}")
    return data

def format_documents(documents):
    """
    문서 리스트를 지정된 형식으로 포맷합니다.
    
    :param documents: 로드된 문서 리스트
    :return: 포맷된 문자열
    """
    formatted_content = ""
    for doc in documents:
        # 파일 경로에서 파일 이름만 추출
        file_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
        content = doc.page_content
        formatted_content += f"*****[{file_name}]*****\n\n{content}\n\n"
    return formatted_content

def save_to_file(content, output_path):
    """
    포맷된 내용을 지정된 파일에 저장합니다.
    
    :param content: 저장할 내용
    :param output_path: 출력 파일 경로
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Result saved to {output_path}")
    except Exception as e:
        print(f"Error saving to file {output_path}: {e}")

def main():
    # 현재 스크립트 이름 기반으로 결과 파일 이름 생성
    script_name = get_current_script_name()
    output_file = f"./06-배진우/2024_겨울특강/과제/code/result/{script_name}_result.txt"
    
    # 텍스트 파일 로드
    documents = load_text_documents(path='./', extensions=['txt', 'md', 'ipynb'], encoding='utf-8')
    
    # 로드된 문서 수 확인
    print(f"총 로드된 문서 수: {len(documents)}개")
    
    if not documents:
        print("로드된 문서가 없습니다. Glob 패턴과 파일 위치를 확인하세요.")
        return
    
    # 문서 포맷팅
    formatted_content = format_documents(documents)
    
    # 결과 저장
    save_to_file(formatted_content, output_file)

if __name__ == "__main__":
    main()
