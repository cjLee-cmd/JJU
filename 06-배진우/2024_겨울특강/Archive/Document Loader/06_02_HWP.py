from langchain_teddynote.document_loaders import HWPLoader

# HWP Loader 객체 생성
loader = HWPLoader("/data/디지털 정부혁신 추진계획.hwp")

# 문서 로드
docs = loader.load()

# 결과 출력
print(docs[0].metadata)