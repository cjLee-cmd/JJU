from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.configurable_fields import ConfigurableField
from dotenv import load_dotenv

def initialize_retrievers(doc_list, k=1):
    """
    Initialize BM25 and FAISS retrievers, and combine them using an EnsembleRetriever.

    Args:
        doc_list (list): List of documents.
        k (int): Number of top results to retrieve.

    Returns:
        EnsembleRetriever: Configured ensemble retriever.
    """
    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever.from_texts(doc_list)
    bm25_retriever.k = k

    # Initialize FAISS retriever
    embedding = OpenAIEmbeddings()
    faiss_vectorstore = FAISS.from_texts(doc_list, embedding)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": k})

    # Combine retrievers with EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
    ).configurable_fields(
        weights=ConfigurableField(
            id="ensemble_weights",
            name="Ensemble Weights",
            description="Ensemble Weights",
        )
    )

    return ensemble_retriever

def retrieve_documents(query, retriever, config):
    """
    Retrieve documents using the provided retriever and configuration.

    Args:
        query (str): Input query.
        retriever (EnsembleRetriever): Configured ensemble retriever.
        config (dict): Configuration parameters for the retriever.

    Returns:
        list: List of retrieved documents.
    """
    return retriever.invoke(query, config=config)

if __name__ == "__main__":

    load_dotenv()

    # Sample document list
    doc_list = [
        "I like apples",
        "I like apple company",
        "I like apple's iphone",
        "Apple is my favorite company",
        "I like apple's ipad",
        "I like apple's macbook",
    ]

    # Configuration for ensemble retriever
    config = {"configurable": {"ensemble_weights": [0.7, 0.3]}}

    # Initialize retrievers
    ensemble_retriever = initialize_retrievers(doc_list, k=1)

    # Perform retrieval
    query = "my favorite fruit is apple"
    docs = retrieve_documents(query, ensemble_retriever, config)

    # Output the retrieved documents
    print(docs)