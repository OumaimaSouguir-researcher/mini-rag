"""
CLI script for ingesting documents into the vector store.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.ingestion.loader import load_docs, load_directory
from app.ingestion.chunker import chunk_docs
from app.embeddings.embedder import get_embedder
from app.vectorstore.faiss_store import build_store


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument(
        "path",
        help="Path to document or directory"
    )
    parser.add_argument(
        "--store-path",
        default="./data/vectorstore",
        help="Path to save vector store"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size in characters"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap in characters"
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model name"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Starting document ingestion...")
    print("=" * 60)
    
    # Load documents
    print(f"\n[1/4] Loading documents from: {args.path}")
    path = Path(args.path)
    if path.is_dir():
        docs = load_directory(str(path))
    else:
        docs = load_docs(str(path))
    
    if not docs:
        print("No documents loaded. Exiting.")
        return
    
    print(f"Loaded {len(docs)} documents")
    
    # Chunk documents
    print(f"\n[2/4] Chunking documents (size={args.chunk_size}, overlap={args.chunk_overlap})")
    chunks = chunk_docs(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    
    # Initialize embedder
    print(f"\n[3/4] Initializing embeddings model: {args.embedding_model}")
    embedder = get_embedder(args.embedding_model)
    
    # Build and save vector store
    print(f"\n[4/4] Building vector store and saving to: {args.store_path}")
    build_store(chunks, embedder, args.store_path)
    
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()