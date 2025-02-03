with open("c:/Users/kyh97/OneDrive - 전주대학교/전주대/2024 2학기/비트캠프/12월27일/appendix-keywords.txt", encoding="utf-8") as f:
  file = f.read()
# with open("C:/Users/kyh97/OneDrive - 전주대학교/전주대/2024 2학기/비트캠프/12월30일일/appendix-keywords.txt") as f:
#   file = f.read()

# pip install -q tiktoken
# pip install -qU spacy
# !python -m spacy download en_core_web_sm --quiet
# !pip install -qU konlpy


from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import SpacyTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_text_splitters import KonlpyTextSplitter
from transformers import GPT2TokenizerFast
import chunk
import warnings

warnings.filterwarnings("ignore")

splitter_tiktoken = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=0,
)

splitter_TokenTextSplitter = TokenTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
)

splitter_Spacy = SpacyTextSplitter(
    chunk_size = 200,
    chunk_overlap=0,
)

splitter_sentenceTransformers = SentenceTransformersTokenTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)

splitter_Konlpy = KonlpyTextSplitter(
    chunk_size=200, 
    chunk_overlap=50
    )

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
splitter_huggingface_gpt2 = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=300,
    chunk_overlap=50,
)


texts_tiktoken             = splitter_tiktoken.split_text(file)
texts_TokenTextSplitter    = splitter_TokenTextSplitter.split_text(file)
texts_Spacy                = splitter_Spacy.split_text(file)
texts_sentenceTransformers = splitter_sentenceTransformers.split_text(file)
texts_Konlpy               = splitter_Konlpy.split_text(file)
texts_huggingface_gpt2     = splitter_huggingface_gpt2.split_text(file)

print(texts_tiktoken[0])
print(texts_TokenTextSplitter[0])
print(texts_Spacy[0])
print(texts_sentenceTransformers[0])
print(texts_Konlpy[0])
print(texts_huggingface_gpt2[0])
