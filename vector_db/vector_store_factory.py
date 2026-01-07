"""
Factory for creating vector stores
"""
import config
from .faiss_store import FAISSStore


class VectorStoreFactory:
    """Factory to create appropriate vector store"""
    
    @classmethod
    def create_store(cls, index_name: str, db_type: str = None) -> FAISSStore:
        """
        Create vector store for an index
        
        Args:
            index_name: Name of the index
            db_type: Type of vector DB (defaults to config)
            
        Returns:
            Vector store instance
        """
        db_type = db_type or config.VECTOR_DB_TYPE
        
        if db_type == "faiss":
            return FAISSStore(index_name)
        else:
            raise ValueError(f"Unsupported vector DB type: {db_type}")

