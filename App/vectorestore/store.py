
"""
FAISS vector store management.
"""

from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS


def build_store(docs: List, embedder, store_path: str):
    """
    Build FAISS vector store from documents and save to disk.
    
    Args:
        docs: List of chunked documents
        embedder: Embeddings instance
        store_path: Path to save the vector store
        
    Returns:
        FAISS vector store instance
    """
    store = FAISS.from_documents(docs, embedder)
    
    # Ensure directory exists
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    
    store.save_local(store_path)
    print(f"Saved vector store to {store_path}")
    
    return store


def load_store(store_path: str, embedder):
    """
    Load FAISS vector store from disk.
    
    Args:
        store_path: Path to the saved vector store
        embedder: Embeddings instance (must match the one used to create store)
        
    Returns:
        FAISS vector store instance
        
    Raises:
        FileNotFoundError: If store does not exist
    """
    if not Path(store_path).exists():
        raise FileNotFoundError(f"Vector store not found at {store_path}")
    
    store = FAISS.load_local(
        store_path, 
        embedder,
        allow_dangerous_deserialization=True
    )
    print(f"Loaded vector store from {store_path}")
    
    return store


def add_documents(store, docs: List):
    """
    Add new documents to existing vector store.
    
    Args:
        store: Existing FAISS store
        docs: New documents to add
    """
    store.add_documents(docs)
    print(f"Added {len(docs)} documents to vector store")