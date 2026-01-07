"""
Factory for creating appropriate chunkers based on document type
"""
from typing import Type
from .base_chunker import BaseChunker
from .policy_chunker import PolicyChunker
from .legal_chunker import LegalChunker
from .technical_chunker import TechnicalChunker
import config


class ChunkingFactory:
    """Factory to create appropriate chunker for document type"""
    
    _chunkers = {
        "policy": PolicyChunker,
        "legal": LegalChunker,
        "technical": TechnicalChunker
    }
    
    @classmethod
    def get_chunker(cls, doc_type: str) -> BaseChunker:
        """
        Get appropriate chunker for document type
        
        Args:
            doc_type: Type of document (policy, legal, technical)
            
        Returns:
            Appropriate chunker instance
        """
        if doc_type not in cls._chunkers:
            raise ValueError(f"Unknown document type: {doc_type}. Must be one of {list(cls._chunkers.keys())}")
        
        chunker_class = cls._chunkers[doc_type]
        chunk_config = config.CHUNK_CONFIG.get(doc_type, {})
        
        return chunker_class(
            chunk_size=chunk_config.get("chunk_size", 500),
            chunk_overlap=chunk_config.get("chunk_overlap", 50)
        )

