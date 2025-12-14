"""
Document loader module for handling different file formats.
Supports PDF, Markdown, and plain text files.
"""

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader


def load_docs(path: str) -> List:
    """
    Load documents from file path based on extension.
    
    Args:
        path: Path to the document file
        
    Returns:
        List of loaded documents
        
    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If file does not exist
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif path.endswith(".md"):
        loader = UnstructuredMarkdownLoader(path)
    elif path.endswith(".txt"):
        loader = TextLoader(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    
    return loader.load()


def load_directory(directory: str, extensions: List[str] = [".pdf", ".md", ".txt"]) -> List:
    """
    Load all documents from a directory.
    
    Args:
        directory: Path to directory
        extensions: List of file extensions to process
        
    Returns:
        List of all loaded documents
    """
    dir_path = Path(directory)
    all_docs = []
    
    for ext in extensions:
        for file_path in dir_path.rglob(f"*{ext}"):
            try:
                docs = load_docs(str(file_path))
                all_docs.extend(docs)
                print(f"Loaded: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return all_docs