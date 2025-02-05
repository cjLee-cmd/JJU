import os
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response
import streamlit as st

@st.cache_resource
def calculate_file_hash(filepath):
    """Calculates the MD5 hash of the given file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buffer = f.read()
        hasher.update(buffer)
    return hasher.hexdigest()

@st.cache_resource
def process_pdf_and_create_db(pdf_filepath):
    """Processes the uploaded PDF file, splits its content, and creates a vector database."""
    print(f"Loading PDF file from: {pdf_filepath}")
    loader = PyMuPDFLoader(pdf_filepath)
    pages = loader.load()

    text_splitter = SemanticChunker(
        OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY")),
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=1.25,
    )

    texts = []
    for page_number, page in enumerate(pages, start=1):
        print(f"Processing page {page_number}...")
        chunks = text_splitter.split_text(page.page_content)
        texts.extend(chunks)

    print(f"Total number of chunks generated: {len(texts)}")

    print("Creating vector database...")
    embeddings_model = OpenAIEmbeddings()
    db = Chroma.from_texts(
        texts,
        embeddings_model,
        collection_name='esg',
        persist_directory='./db/chromadb',
        collection_metadata={'hnsw:space': 'cosine'},
    )
    print("Vector database created successfully.")
    return db

def query_database(db, query):
    """Queries the vector database using Max Marginal Relevance search."""
    print("Querying the database...")
    mmr_docs = db.max_marginal_relevance_search(query, k=5, fetch_k=20)
    print(f"Number of relevant documents found: {len(mmr_docs)}")
    return mmr_docs

def generate_response(query, mmr_docs):
    """Generates a response for the given query using the retrieved documents."""
    question = {
        "instruction": query,
        "mmr_docs": mmr_docs
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Analyze the following content and answer the question. Answer in Korean."),
            ("human", "{instruction}\n{mmr_docs}"),
        ]
    )

    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o",
    )

    chain = prompt | llm
    print("Generating response...")
    answer = chain.stream(question)

    st.markdown(f"<p style='font-size:20px;'>{query}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:18px;'>Response: {stream_response(answer, return_output=True)}</p>", unsafe_allow_html=True)

def main_streamlit():
    load_dotenv()

    st.set_page_config(page_title="Semantic Analysis", page_icon="🔬", layout="wide")

    st.sidebar.title("Options")
    st.sidebar.write("Upload a file and enter your query to analyze it.")

    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    query = st.sidebar.text_input("Enter your query:")
    query_button = st.sidebar.button("Send Query")

    st.title("🔬 Semantic Analysis with LangChain")
    st.markdown("""
    This application allows you to upload a PDF, analyze its content, and retrieve information using natural language queries.
    
    **Features:**
    - Semantic chunking for better document understanding.
    - Vector database for fast and accurate search.
    - Responses in Korean with typing animation.
    """)

    if uploaded_file:
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        file_hash = calculate_file_hash(temp_file_path)
        db = process_pdf_and_create_db(temp_file_path)

        st.session_state["db"] = db  # Store the database in session state
        st.session_state["file_hash"] = file_hash

    if query_button:
        if "db" not in st.session_state:
            st.sidebar.error("Please upload a file to create the database first.")
        else:
            try:
                db = st.session_state["db"]
                mmr_docs = query_database(db, query)
                generate_response(query, mmr_docs)
            except Exception as e:
                st.sidebar.error(f"An error occurred during querying: {e}")

if __name__ == "__main__":
    main_streamlit()
