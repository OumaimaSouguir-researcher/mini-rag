"""
Embedding generation module using sentence-transformers.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initialize the embedding model.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Embeddings instance
    """
    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print(f"Loaded embeddings model: {model_name}")
    return embedder


def get_fast_embedder():
    """
    Get a lighter, faster embedding model for quick testing.
    
    Returns:
        Embeddings instance with lighter model
    """
    return get_embedder("sentence-transformers/paraphrase-MiniLM-L3-v2")


def embed_query(embedder, query: str):
    """
    Embed a single query string.
    
    Args:
        embedder: Embeddings instance
        query: Query text
        
    Returns:
        Embedding vector
    """
    return embedder.embed_query(query)