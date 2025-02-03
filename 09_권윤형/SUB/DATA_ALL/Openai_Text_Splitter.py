OPENAI_API_KEY = 'API_KEY'

with open("c:/Users/kyh97/OneDrive - 전주대학교/전주대/2024 2학기/비트캠프/12월27일/appendix-keywords.txt", encoding="utf-8") as f:
  file = f.read()

# pip install -q langchain_experimental
# pip install -q langchain_openai

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

text_splitter = SemanticChunker(OpenAIEmbeddings(api_key=OPENAI_API_KEY))

text_splitter = SemanticChunker(
    OpenAIEmbeddings(api_key=OPENAI_API_KEY),

    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=70,
)

docs = text_splitter.create_documents([file])
for i, doc in enumerate(docs[:5]):
    print(f"[Chunk {i}]", end='\n\n')
    print(doc.page_content)
    print('==='*20)

text_splitter = SemanticChunker(
    OpenAIEmbeddings(api_key=OPENAI_API_KEY),

    breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=1.25,
)

docs = text_splitter.create_documents([file])
for i, doc in enumerate(docs[:5]):
    print(f"[Chunk {i}]", end='\n\n')
    print(doc.page_content)
    print('==='*20)