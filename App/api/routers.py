"""
FastAPI routes for RAG API.
"""

from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import (
    QueryRequest, QueryResponse, ContextItem,
    AskRequest, AskResponse, HealthResponse
)
from app.rag.retriever import retrieve, retrieve_with_scores
from app.core.config import get_settings

router = APIRouter()


def get_vector_store():
    """Dependency to get vector store instance."""
    from app.main import app
    if not hasattr(app.state, 'vector_store') or app.state.vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return app.state.vector_store


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and component availability."""
    from app.main import app
    
    vectorstore_loaded = hasattr(app.state, 'vector_store') and app.state.vector_store is not None
    llm_available = hasattr(app.state, 'llm') and app.state.llm is not None
    
    return HealthResponse(
        status="healthy",
        vectorstore_loaded=vectorstore_loaded,
        llm_available=llm_available
    )


@router.post("/query", response_model=QueryResponse)
async def query_vectorstore(
    request: QueryRequest,
    store=Depends(get_vector_store)
):
    """
    Query the vector store and return relevant contexts.
    
    Args:
        request: Query request with query string and k value
        
    Returns:
        QueryResponse with relevant contexts
    """
    try:
        docs_with_scores = retrieve_with_scores(request.query, store, k=request.k)
        
        contexts = [
            ContextItem(
                content=doc.page_content,
                metadata=doc.metadata,
                score=float(score)
            )
            for doc, score in docs_with_scores
        ]
        
        return QueryResponse(
            query=request.query,
            contexts=contexts,
            count=len(contexts)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    store=Depends(get_vector_store)
):
    """
    Ask a question and get an LLM-generated answer based on retrieved contexts.
    
    Args:
        request: Ask request with question
        
    Returns:
        AskResponse with answer and contexts
    """
    from app.main import app
    
    if not hasattr(app.state, 'rag_chain') or app.state.rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    
    try:
        # Retrieve contexts
        docs_with_scores = retrieve_with_scores(request.question, store, k=request.k)
        
        contexts = [
            ContextItem(
                content=doc.page_content,
                metadata=doc.metadata,
                score=float(score)
            )
            for doc, score in docs_with_scores
        ]
        
        # Generate answer
        answer = app.state.rag_chain.run(request.question)
        
        return AskResponse(
            question=request.question,
            answer=answer,
            contexts=contexts
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")