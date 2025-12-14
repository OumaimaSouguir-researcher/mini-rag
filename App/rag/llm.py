"""
Local LLM integration module using Ollama.
"""

from langchain_community.llms import Ollama
from app.core.config import get_settings

settings = get_settings()


def get_llm(model: str = None, temperature: float = 0.1):
    """
    Initialize Ollama LLM instance.
    
    Args:
        model: Model name (defaults to settings)
        temperature: Sampling temperature (0 = deterministic, 1 = creative)
        
    Returns:
        Ollama LLM instance
    """
    if model is None:
        model = settings.ollama_model
    
    llm = Ollama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=temperature
    )
    
    print(f"Initialized Ollama LLM: {model}")
    return llm


def check_ollama_available() -> bool:
    """
    Check if Ollama server is available.
    
    Returns:
        True if Ollama is reachable, False otherwise
    """
    try:
        llm = get_llm()
        # Simple test query
        llm.invoke("test")
        return True
    except Exception as e:
        print(f"Ollama not available: {e}")
        return False


def list_available_models():
    """
    List available Ollama models.
    
    Note: This requires ollama CLI or API access.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except Exception as e:
        return f"Could not list models: {e}"