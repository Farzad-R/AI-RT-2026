from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRAGEngine(ABC):
    """
    Abstract Base Class for RAG Engines.
    
    This enforces a standard interface for any Retrieval Strategy 
    (Dense, Sparse, Multi-Vector, Hybrid, Graph) so the UI 
    can switch between them without changing frontend code.
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize models and database connections using a config dictionary.
        
        Args:
            config: Dictionary containing API keys, model names, and DB paths.
        """
        pass

    @abstractmethod
    def index(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Ingest documents into the vector database.
        
        Args:
            documents: List of text strings to index.
            metadata: List of dictionaries containing metadata for each doc.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search the database.
        
        Args:
            query: The search query string.
            top_k: Number of results to return.

        Returns:
            List of dictionaries containing:
            {'text': str, 'score': float, 'metadata': dict}
        """
        pass

    @abstractmethod
    def answer(self, query: str, context: List[Dict]) -> str:
        """
        Generate an answer using an LLM based on retrieved context.
        
        Args:
            query: User question.
            context: List of retrieved chunks from .retrieve()
            
        Returns:
            The generated string response.
        """
        pass