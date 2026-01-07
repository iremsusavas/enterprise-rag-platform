"""
Policy Document Chunker
Uses section-based chunking for policy documents
"""
from typing import List, Dict
import re
from .base_chunker import BaseChunker


class PolicyChunker(BaseChunker):
    """
    Chunks policy documents by sections
    Policy documents typically have clear section headers
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        # Pattern to match section headers (e.g., "1. Section Title", "Section 2:", etc.)
        self.section_pattern = re.compile(
            r'^(?:\d+\.?\s+)?[A-Z][^.!?]*[:\-]?\s*$',
            re.MULTILINE
        )
    
    def chunk(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Chunk policy document by sections
        
        Strategy:
        - Identify section boundaries
        - Keep sections together when possible
        - Split large sections by size
        """
        # Find section boundaries
        sections = self._split_by_sections(text)
        
        chunks = []
        for section_idx, section in enumerate(sections):
            if len(section) <= self.chunk_size:
                # Section fits in one chunk
                chunks.append(self._add_metadata(
                    section, 
                    len(chunks), 
                    {**metadata, "section_index": section_idx}
                ))
            else:
                # Split large section by size
                section_chunks = self._split_by_size(section, {**metadata, "section_index": section_idx})
                chunks.extend(section_chunks)
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by section headers"""
        lines = text.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            # Check if line is a section header
            if self.section_pattern.match(line.strip()):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add last section
        if current_section:
            sections.append('\n'.join(current_section))
        
        # If no sections found, return entire text as one section
        if not sections:
            return [text]
        
        return sections

