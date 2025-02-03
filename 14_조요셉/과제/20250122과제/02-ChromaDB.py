import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_upstage import UpstageDocumentParseLoader # 위키독스 따라가면 안됨.

load_dotenv()

def prossed_document(file_path):
    loader = UpstageDocumentParseLoader(file_path=file_path, output_format="html", split="element", ocr="force", exclude=["header", "footer"],)
    docs = loader.load()

    return docs

def create_db(DB_PATH, documents, embedding=OpenAIEmbeddings(model="text-embedding-3-small"), name="db"):
    db = Chroma.from_documents(
    documents=documents, persist_directory=DB_PATH, embedding=embedding, collection_name=name
    )

def add_data(db, documents):
    db.add_texts(documents)

# def dorp_data():

def select_data(db):
    return db.got()

DB_PATH = "./chroma_db"
name = "my_db"
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
file_path = "과제/20250122과제/data/Aligning Instruction Tuning with Pre-training.pdf"

documents = prossed_document(file_path)

create_db(DB_PATH=DB_PATH, documents=documents, embedding=embedding, name=name)

db = Chroma(persist_directory=DB_PATH, embedding_function=embedding, collection_name=name,)

print(select_data(db))
