# 현재 진행상황
파일 형식에 따른 Loader 사용 
Loader로 로드된 데이터를 Splitter 라이브러리에 적용 중
splitter_pdf 에서 RecursiveCharacterTextSplitter만 구현

# 앞으로 구현해야 하는 것
나머지 splitter 구현 해야됨.
split된 데이터를 llm에 넣어서 스코어 뽑아서 csv에 저장하는 코드 구현
추가로 chunk size, overlap 변경하면서 스코어 csv에 추가하는 코드 구현