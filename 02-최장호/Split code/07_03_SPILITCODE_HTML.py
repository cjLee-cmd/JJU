%pip install -qU langchain-text-splitters

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)

html_text = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>비디오 공유 사이트</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: #f9f9f9;
        }
        header {
            background: #cc0000;
            color: white;
            padding: 10px 20px;
            text-align: center;
        }
        main {
            margin: 20px;
        }
        .video {
            background: white;
            margin: 10px 0;
            padding: 20px;
            border: 1px solid #ddd;
            text-align: center; /* 비디오를 중앙 정렬합니다 */
        }
        .video h2 {
            color: #333;
        }
        .video iframe {
            width: 560px; /* 비디오 폭을 560px로 조절 */
            height: 315px; /* 비디오 높이를 315px로 조절 */
        }
        footer {
            text-align: center;
            padding: 20px 0;
            background: #333;
            color: white;
        }
    </style>
</head>
<body>
    <header>
        <h1>AI Agent Video share</h1>
    </header>
    <main>
        <div class="video">
            <h2>What are AI Agents?</h2>
            <iframe src="https://www.youtube.com/embed/F8NKVhkZZWI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </div>
        <div class="video">
            <h2>Andrew Ng Explores The Rise Of AI Agents And Agentic Reasoning | BUILD 2024 Keynote</h2>
            <iframe src="https://www.youtube.com/embed/KrRD7r7y7NY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </div>
    </main>
    <footer>
        <p>© 2024 비디오 공유 사이트. 모든 권리 보유.</p>
    </footer>
</body>
</html>
"""

html_splitter = RecursiveCharacterTextSplitter.from_language(
    # HTML 언어를 사용하여 텍스트 분할기 생성
    language=Language.HTML,
    # 청크 크기를 60으로 설정
    chunk_size=60,
    # 청크 간 중복되는 부분이 없도록 설정
    chunk_overlap=0,
)
# 주어진 HTML 텍스트를 분할하여 문서 생성
html_docs = html_splitter.create_documents([html_text])
# 생성된 문서 출력
html_docs