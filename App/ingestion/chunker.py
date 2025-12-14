"""
Text chunking module for splitting documents into optimal sizes.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_docs(docs: List, chunk_size: int = 800, chunk_overlap: int = 150) -> List:
    """
    Split documents into smaller chunks for embedding.
    
    Args:
        docs: List of documents to chunk
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of chunked documents
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} documents")
    
    return chunks


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """
    Split raw text into chunks.
    
    Args:
        text: Raw text string
        chunk_size: Target size of each chunk
        chunk_overlap: Number of overlapping characters
        
    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    return splitter.split_text(text)