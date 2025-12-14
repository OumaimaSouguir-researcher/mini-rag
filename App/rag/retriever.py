"""
Retrieval module for querying the vector store.
"""

from typing import List
from langchain_community.vectorstores import FAISS


def load_store(store_path: str, embedder):
    """
    Load FAISS vector store from disk.
    
    Args:
        store_path: Path to saved vector store
        embedder: Embeddings instance
        
    Returns:
        FAISS store instance
    """
    return FAISS.load_local(
        store_path,
        embedder,
        allow_dangerous_deserialization=True
    )


def retrieve(query: str, store, k: int = 4) -> List:
    """
    Retrieve k most similar documents to the query.
    
    Args:
        query: Query string
        store: FAISS vector store
        k: Number of documents to retrieve
        
    Returns:
        List of relevant documents
    """
    docs = store.similarity_search(query, k=k)
    return docs


def retrieve_with_scores(query: str, store, k: int = 4):
    """
    Retrieve documents with similarity scores.
    
    Args:
        query: Query string
        store: FAISS vector store
        k: Number of documents to retrieve
        
    Returns:
        List of tuples (document, score)
    """
    docs_with_scores = store.similarity_search_with_score(query, k=k)
    return docs_with_scores


def retrieve_as_retriever(store, k: int = 4):
    """
    Get retriever interface from vector store.
    
    Args:
        store: FAISS vector store
        k: Number of documents to retrieve
        
    Returns:
        Retriever instance
    """
    return store.as_retriever(search_kwargs={"k": k})