"""
Pydantic models for API request/response validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for querying the vector store."""
    query: str = Field(..., description="Query string", min_length=1)
    k: int = Field(4, description="Number of documents to retrieve", ge=1, le=20)


class ContextItem(BaseModel):
    """Single context item from retrieval."""
    content: str
    metadata: Optional[dict] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    contexts: List[ContextItem]
    count: int


class AskRequest(BaseModel):
    """Request model for RAG question answering."""
    question: str = Field(..., description="Question to answer", min_length=1)
    k: int = Field(4, description="Number of context documents", ge=1, le=20)


class AskResponse(BaseModel):
    """Response model for ask endpoint."""
    question: str
    answer: str
    contexts: List[ContextItem]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vectorstore_loaded: bool
    llm_available: bool