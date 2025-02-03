import os
import nest_asyncio

LLAMA_CLOUD_API_KEY = 'LLAMA_CLOUD_API_KEY'
nest_asyncio.apply()

from llama_parse import LlamaParse

OPENAI_API_KEY = 'OPENAI_API_KEY'

documents = LlamaParse(
    use_vendor_mulitmodal_model = True,
    vendor_multimodal_model_name = 'openai-gpt4o',
    vendor_multimodal_api = OPENAI_API_KEY,
    api_key=LLAMA_CLOUD_API_KEY,
    result_type='markdown',
    language='ko',
    #skip_diagonal_text = True,
    #page_seperator='\n======================\n',
)

parsed_docs = documents.load_data(file_path='/content/data/ko_pdf.pdf')

docs = [doc.to_langchain_format() for doc in parsed_docs]

# print(docs)
# print(len(docs))
print(docs[0].page_content)