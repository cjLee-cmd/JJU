import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import logging
from datetime import datetime

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_vector_database(chunks):
    logging.info(f"Creating vector database...")

    persist_directory = './SWEngineer/db/chromadb'
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        logging.info(f"Created directory: {persist_directory}")

    try:
        embeddings_model = OpenAIEmbeddings()
        logging.info(f"Initialized OpenAI embeddings model.")

        # Chroma 데이터베이스 생성
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            collection_name='esg',
            persist_directory=persist_directory,
            collection_metadata={'hnsw:space': 'cosine'},
        )

        logging.info(f"Vector database created successfully.")

        return db

    except Exception as e:
        logging.error(f"Error during vector database creation: {e}")
        return None

def query_database(db, query):
    logging.info(f"Querying the vector database...")
    try:
        mmr_docs = db.max_marginal_relevance_search(query, k=20, fetch_k=100)
        logging.info(f"Query returned {len(mmr_docs)} documents. \n docs : {mmr_docs}")
        return [docs.page_content for docs in mmr_docs]

    except Exception as e:
        logging.error(f"Error during querying: {e}")
        raise
