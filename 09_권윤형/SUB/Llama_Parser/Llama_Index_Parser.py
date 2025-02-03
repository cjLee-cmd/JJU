LLAMA_CLOUD_API_KEY = 'API_KEY'

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

parser = LlamaParse(
    result_type = 'markdown',
    num_workers=8,
    verbose=True,
    language='ko',
    api_key=LLAMA_CLOUD_API_KEY,
)

file_extractor = {'.pdf':parser}

documents = SimpleDirectoryReader(
    input_files = ['/content/data/Attention is all you need.pdf'],
    file_extractor = file_extraSctor,
).load_data()

print(len(documents))
print(documents)

# Llama Index -> Langchain Document 변환
docs = [doc.to_langchain_format() for doc in documents]
print(docs)
print(len(docs))
print(docs[0].page_content)

#-----------------------------------------------------
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

parsed_docs = documents.load_data(file_path='/content/data/Attention is all you need.pdf')

docs = [doc.to_langchain_format() for doc in parsed_docs]

print(docs)
print(len(docs))
print(docs[0].page_content)

# Parsing Instruction

parsing_instruction = (
    '너가 알아서 parsing 해.'
)

parser = LlamaParse(
    use_vendor_mulitmodal_model = True,
    vendor_multimodal_model_name = 'openai-gpt4o',
    vendor_multimodal_api = OPENAI_API_KEY,
    api_key=LLAMA_CLOUD_API_KEY,
    result_type='markdown',
    language='eng',
    parsing_instruction=parsing_instruction,
    #skip_diagonal_text = True,
    #page_seperator='\n======================\n',
)

parsed_docs = documents.load_data(file_path='/content/data/Attention is all you need.pdf')

docs = [doc.to_langchain_format() for doc in parsed_docs]

print(docs)
print(len(docs))
print(docs[0].page_content)