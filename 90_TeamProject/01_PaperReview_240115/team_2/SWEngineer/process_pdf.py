import os
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Separate function for validation
def validate_pdf_file(pdf_filepath):
    if not os.path.isfile(pdf_filepath):
        raise FileNotFoundError(f"File not found: {pdf_filepath}")
    if not pdf_filepath.lower().endswith('.pdf'):
        raise ValueError("File must be a PDF.")

# Separate function for extracting text from PDF
def extract_text_from_pdf(pdf_filepath):
    loader = PyMuPDFLoader(pdf_filepath)
    pages = loader.load()
    logging.info(f"Loaded {len(pages)} pages from the PDF.")
    return [(page.page_content,
            page.metadata['page'],
            page.metadata['total_pages'],
            page.metadata['author'],
            page.metadata['subject'],
            page.metadata['keywords'],
            page.metadata['creationDate'],
            page.metadata['modDate']) for page in pages]

def split_text_into_chunks_parallel(texts, text_splitter):
    all_chunks = []
    logging.info("Starting text splitting into chunks.")

    with ThreadPoolExecutor() as executor:
        results = executor.map(text_splitter.split_text, [text[0] for text in texts])

        for idx, chunks in enumerate(results):
            logging.info(f"Processing text from page {texts[idx][1]} (Page {idx + 1}).")
            for chunk_idx, chunk in enumerate(chunks):
                # Add metadata to each chunk
                chunk_metadata = Document(
                    page_content = chunk,
                    metadata = {
                        "chunk_id": f"chunk_{idx + 1}_{chunk_idx + 1}",  # Unique ID for each chunk
                        "page_number": texts[idx][1],  # Page number where the chunk came from
                        "total_pages": texts[idx][2],  # Total pages in the document
                        "author": texts[idx][3],  # Author from page metadata
                        "subject": texts[idx][4],  # Subject from page metadata
                        "keywords": texts[idx][5],  # Keywords from page metadata
                        "creationDate": texts[idx][6],  # Creation date from page metadata
                        "modDate": texts[idx][7],  # Modification date from page metadata
                        "chunk_length": len(chunk),  # Length of the chunk
                        "timestamp": datetime.now().isoformat()  # Timestamp for when the chunk is created
                    },
                    id = chunk_idx + 1
                )
                all_chunks.append(chunk_metadata)
                logging.info(f"Added chunk {chunk_metadata.id} (Length: {chunk_metadata.metadata['chunk_length']} characters).")

    logging.info(f"Completed splitting into {len(all_chunks)} chunks.")
    return all_chunks

# Main processing function
def process_pdf(pdf_filepath, api_key=None, threshold_type="standard_deviation", threshold_amount=1.25):
    logging.info(f"Starting PDF processing for file: {pdf_filepath}")

    # Validate file
    validate_pdf_file(pdf_filepath)

    # Check API key
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is not provided. Set 'OPENAI_API_KEY' or pass it as an argument.")

    try:
        logging.info(f"OPENAI API KEY : {api_key}")
        # Extract text from PDF
        texts = extract_text_from_pdf(pdf_filepath)

        # Initialize text splitter
        text_splitter = SemanticChunker(
            OpenAIEmbeddings(model="text-embedding-3-large"),
            breakpoint_threshold_type=threshold_type,
            breakpoint_threshold_amount=threshold_amount,
        )
        logging.info("Initialized semantic chunker.")

        # Process texts in parallel
        chunks = split_text_into_chunks_parallel(texts, text_splitter)
        logging.info(f"Processing complete. Generated {len(chunks)} chunks from the PDF.")

        return chunks

    except Exception as e:
        logging.error(f"Error during PDF processing: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    setup_logging()
    try:
        pdf_path = "/content/data/Agentic Search-Enhanced.pdf"  # Replace with your PDF path
        chunks = process_pdf(pdf_path)
        print(f"Generated {len(chunks)} chunks.")
        print(chunks)
    except Exception as e:
        print(f"Failed to process PDF: {e}")