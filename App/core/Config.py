"""
Configuration management using environment variables.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # App settings
    app_name: str = "Mini-RAG"
    debug: bool = False
    
    # Vector store settings
    vectorstore_path: str = "./data/vectorstore"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    
    # Retrieval settings
    default_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 150
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings instance
    """
    return Settings()