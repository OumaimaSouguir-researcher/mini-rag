"""
FastAPI application entrypoint with LLM integration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.api.routes import router
from app.embeddings.embedder import get_embedder
from app.vectorstore.faiss_store import load_store
from app.rag.llm import get_llm, check_ollama_available
from app.rag.chain import build_chain

settings = get_settings()
setup_logging(level="INFO" if not settings.debug else "DEBUG")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI app.
    Loads vector store and LLM on startup.
    """
    logger.info("Starting Mini-RAG API...")
    
    # Load embeddings model
    try:
        logger.info(f"Loading embeddings model: {settings.embedding_model}")
        embedder = get_embedder(settings.embedding_model)
        app.state.embedder = embedder
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        app.state.embedder = None
    
    # Load vector store
    try:
        logger.info(f"Loading vector store from: {settings.vectorstore_path}")
        store = load_store(settings.vectorstore_path, app.state.embedder)
        app.state.vector_store = store
        logger.info("Vector store loaded successfully")
    except Exception as e:
        logger.warning(f"Vector store not loaded: {e}")
        logger.warning("Run ingestion script first: python scripts/ingest.py <path>")
        app.state.vector_store = None
    
    # Load LLM
    if check_ollama_available():
        try:
            logger.info(f"Loading Ollama LLM: {settings.ollama_model}")
            llm = get_llm()
            app.state.llm = llm
            
            # Build RAG chain if vector store is available
            if app.state.vector_store is not None:
                logger.info("Building RAG chain...")
                rag_chain = build_chain(llm, app.state.vector_store)
                app.state.rag_chain = rag_chain
                logger.info("RAG chain ready")
            else:
                app.state.rag_chain = None
                logger.warning("RAG chain not built (vector store unavailable)")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            app.state.llm = None
            app.state.rag_chain = None
    else:
        logger.warning("Ollama not available. Install with: curl -fsSL https://ollama.com/install.sh | sh")
        logger.warning(f"Then run: ollama pull {settings.ollama_model}")
        app.state.llm = None
        app.state.rag_chain = None
    
    logger.info("API ready to serve requests")
    
    yield
    
    logger.info("Shutting down Mini-RAG API...")


app = FastAPI(
    title=settings.app_name,
    description="Local RAG API for private document Q&A",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Mini-RAG API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )