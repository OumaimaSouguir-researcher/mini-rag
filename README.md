# Mini-RAG: Local-First RAG System

A production-ready, local-first Retrieval-Augmented Generation (RAG) system for private document Q&A. No cloud dependencies, no API keys, just your documents and your hardware.

## Features

- 🔒 **Fully Local**: All processing happens on your machine
- ⚡ **Fast**: FAISS-powered vector similarity search
- 🎯 **Accurate**: Sentence-transformers embeddings + local LLM
- 🛠️ **Production-Ready**: FastAPI backend with proper error handling
- 🐳 **Docker Support**: One command deployment
- 📚 **Multi-Format**: PDF, Markdown, and plain text support

## Quick Start

### Prerequisites

- Python 3.10+
- Poetry
- Ollama (for local LLM)
- Poppler (for PDF processing)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd mini-rag

# Install dependencies
poetry install

# Copy environment file
cp .env.example .env

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3

# Install poppler (for PDF support)
sudo apt install poppler-utils  # Ubuntu/Debian
brew install poppler            # macOS
```

### Usage

#### 1. Ingest Documents

```bash
# Ingest a single document
poetry run python scripts/ingest.py data/raw/your-document.pdf

# Ingest a directory
poetry run python scripts/ingest.py data/raw/
```

#### 2. Start the API

```bash
poetry run uvicorn app.main:app --reload
```

#### 3. Query Your Documents

```bash
# Query for contexts
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "k": 4}'

# Ask questions (RAG)
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the key points", "k": 4}'
```

## Docker Deployment

```bash
# Build and start services
docker-compose -f docker/docker-compose.yml up --build

# Pull Ollama model (first time only)
docker exec -it mini-rag-ollama ollama pull llama3

# Ingest documents
docker exec -it mini-rag-api python scripts/ingest.py /app/data/raw/
```

## Architecture

```
Documents → Ingestion → Embeddings → Vector Store → Retrieval → LLM → Answer
```

### Components

- **Ingestion Service**: Loads and chunks documents
- **Embeddings Generator**: sentence-transformers models
- **Vector Store**: FAISS for similarity search
- **Retrieval API**: FastAPI endpoints
- **Local LLM**: Ollama (llama3, mistral, etc.)

## Project Structure

```
mini-rag/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── api/routes.py        # API endpoints
│   ├── core/                # Config & logging
│   ├── ingestion/           # Document loading & chunking
│   ├── embeddings/          # Embedding generation
│   ├── vectorstore/         # FAISS operations
│   ├── rag/                 # Retrieval & LLM chain
│   └── models/schemas.py    # Pydantic models
├── data/                    # Data directory
├── scripts/                 # CLI scripts
├── docker/                  # Docker files
└── pyproject.toml          # Dependencies
```

## Configuration

Edit `.env` or set environment variables:

```bash
VECTORSTORE_PATH=./data/vectorstore
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_K=4
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

## API Endpoints

### Health Check
```
GET /api/v1/health
```

### Query Vector Store
```
POST /api/v1/query
{
  "query": "your question",
  "k": 4
}
```

### Ask Question (RAG)
```
POST /api/v1/ask
{
  "question": "your question",
  "k": 4
}
```

## Development

```bash
# Run with auto-reload
poetry run uvicorn app.main:app --reload

# Format code
poetry run black app/

# Lint
poetry run ruff check app/

# Type check
poetry run mypy app/
```

## Troubleshooting

### Vector store not found
```bash
# Run ingestion first
poetry run python scripts/ingest.py data/raw/
```

### Ollama not available
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3

# Check status
ollama list
```

### PDF loading errors
```bash
# Install poppler
sudo apt install poppler-utils  # Ubuntu/Debian
brew install poppler            # macOS
```

## Next Steps

- [ ] Add semantic chunking
- [ ] Implement metadata filters
- [ ] Add reranking with cross-encoders
- [ ] Enable streaming responses
- [ ] Create evaluation harness
- [ ] Implement hybrid search (BM25 + vectors)

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.