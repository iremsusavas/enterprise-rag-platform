"""
Legal Document Chunker
Uses semantic chunking with larger chunks for legal documents
Legal documents require context preservation
"""
from typing import List, Dict
import re
from .base_chunker import BaseChunker


class LegalChunker(BaseChunker):
    """
    Chunks legal documents with semantic awareness
    Legal documents need larger chunks to preserve context
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        # Legal document patterns
        self.paragraph_pattern = re.compile(r'\n\s*\n')  # Double newline = paragraph break
        self.clause_pattern = re.compile(r'^\s*(?:WHEREAS|THEREFORE|NOW THEREFORE|ARTICLE|SECTION)', re.MULTILINE | re.IGNORECASE)
    
    def chunk(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Chunk legal document semantically
        
        Strategy:
        - Preserve paragraphs together
        - Respect clause boundaries
        - Use larger chunks with significant overlap
        """
        # First, try to split by clauses
        clauses = self._split_by_clauses(text)
        
        chunks = []
        current_chunk_parts = []
        current_size = 0
        
        for clause in clauses:
            clause_size = len(clause)
            
            # If adding this clause would exceed chunk size, finalize current chunk
            if current_size + clause_size > self.chunk_size and current_chunk_parts:
                chunk_text = '\n\n'.join(current_chunk_parts)
                chunks.append(self._add_metadata(
                    chunk_text,
                    len(chunks),
                    metadata
                ))
                
                # Overlap: keep last part of previous chunk
                if self.chunk_overlap > 0 and current_chunk_parts:
                    overlap_text = current_chunk_parts[-1]
                    if len(overlap_text) > self.chunk_overlap:
                        # Take last portion for overlap
                        overlap_words = overlap_text.split()
                        overlap_count = int(self.chunk_overlap / 10)
                        overlap_text = ' '.join(overlap_words[-overlap_count:])
                        current_chunk_parts = [overlap_text]
                        current_size = len(overlap_text)
                    else:
                        current_chunk_parts = [overlap_text]
                        current_size = len(overlap_text)
                else:
                    current_chunk_parts = []
                    current_size = 0
            
            # If clause itself is too large, split it
            if clause_size > self.chunk_size:
                clause_chunks = self._split_by_size(clause, metadata)
                chunks.extend(clause_chunks)
                current_chunk_parts = []
                current_size = 0
            else:
                current_chunk_parts.append(clause)
                current_size += clause_size + 2  # +2 for '\n\n'
        
        # Add remaining chunk
        if current_chunk_parts:
            chunk_text = '\n\n'.join(current_chunk_parts)
            chunks.append(self._add_metadata(chunk_text, len(chunks), metadata))
        
        return chunks
    
    def _split_by_clauses(self, text: str) -> List[str]:
        """Split text by legal clause markers"""
        clauses = self.clause_pattern.split(text)
        if len(clauses) > 1:
            # Re-add the clause markers
            matches = list(self.clause_pattern.finditer(text))
            result = []
            for i, clause in enumerate(clauses):
                if i == 0:
                    result.append(clause)
                else:
                    # Prepend the matched clause marker
                    marker = matches[i-1].group()
                    result.append(marker + clause)
            return result
        else:
            # No clauses found, split by paragraphs
            return self.paragraph_pattern.split(text)

