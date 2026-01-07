"""
Base chunker class with common functionality
"""
from abc import ABC, abstractmethod
from typing import List, Dict
import re


class BaseChunker(ABC):
    """Base class for all chunkers"""
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Chunk text into smaller pieces
        
        Args:
            text: Text to chunk
            metadata: Original document metadata
            
        Returns:
            List of chunk dictionaries with 'content' and 'metadata'
        """
        pass
    
    def _add_metadata(self, chunk_text: str, chunk_idx: int, original_metadata: Dict) -> Dict:
        """Add metadata to a chunk"""
        return {
            "content": chunk_text,
            "metadata": {
                **original_metadata,
                "chunk_index": chunk_idx,
                "chunk_size": len(chunk_text)
            }
        }
    
    def _split_by_size(self, text: str, metadata: Dict) -> List[Dict]:
        """Simple size-based chunking with overlap"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(self._add_metadata(chunk_text, len(chunks), metadata))
                
                # Handle overlap
                overlap_words = int(self.chunk_overlap / 10)  # Approximate word count
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                current_size = sum(len(w) + 1 for w in current_chunk)
            
            current_chunk.append(word)
            current_size += word_size
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._add_metadata(chunk_text, len(chunks), metadata))
        
        return chunks

