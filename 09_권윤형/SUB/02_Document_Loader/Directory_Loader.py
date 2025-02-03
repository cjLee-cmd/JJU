from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PythonLoader
from langchain_community.document_loaders import TextLoader

loader = DirectoryLoader('./', glob = '**/*.md')                            # **/ : 하위폴더 재귀적으로 체크
loader = DirectoryLoader('./', glob = '**/*.md', show_progress = True)      # 진행상황 표시
loader = DirectoryLoader('./', glob = '**/*.md', use_multithreading = True) # 멀티스레딩 사용
loader = DirectoryLoader('./', glob = '**/*.md', loader_cls = TextLoader)   # TextLoader 사용
loader = DirectoryLoader('./', glob = '**/*.py', loader_cls = PythonLoader) # PythonLoader 사용

docs = loader.load()

#print(len(docs))
#print(docs[0].page_content)
print(docs[2])