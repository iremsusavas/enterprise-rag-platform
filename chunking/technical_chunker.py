"""
Technical Document Chunker
Uses function/heading-based chunking for technical documents
"""
from typing import List, Dict
import re
from .base_chunker import BaseChunker


class TechnicalChunker(BaseChunker):
    """
    Chunks technical documents by functions, classes, and headings
    Technical documents have code-like structures
    """
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        super().__init__(chunk_size, chunk_overlap)
        # Patterns for technical document structures
        self.heading_pattern = re.compile(r'^#{1,6}\s+.+$', re.MULTILINE)  # Markdown headings
        self.function_pattern = re.compile(r'^(?:def|function|class|interface|type)\s+\w+', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    
    def chunk(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Chunk technical document by functions/headings
        
        Strategy:
        - Preserve code blocks together
        - Keep functions/classes together
        - Respect heading boundaries
        """
        # First, extract and preserve code blocks
        code_blocks = {}
        text_without_code = text
        for i, match in enumerate(self.code_block_pattern.finditer(text)):
            placeholder = f"__CODE_BLOCK_{i}__"
            code_blocks[placeholder] = match.group()
            text_without_code = text_without_code.replace(match.group(), placeholder)
        
        # Split by headings or functions
        sections = self._split_by_structure(text_without_code)
        
        chunks = []
        for section_idx, section in enumerate(sections):
            # Restore code blocks
            for placeholder, code in code_blocks.items():
                section = section.replace(placeholder, code)
            
            if len(section) <= self.chunk_size:
                chunks.append(self._add_metadata(
                    section,
                    len(chunks),
                    {**metadata, "section_index": section_idx}
                ))
            else:
                # Split large section
                section_chunks = self._split_by_size(section, {**metadata, "section_index": section_idx})
                chunks.extend(section_chunks)
        
        return chunks
    
    def _split_by_structure(self, text: str) -> List[str]:
        """Split text by headings or function definitions"""
        # Try heading-based first
        heading_matches = list(self.heading_pattern.finditer(text))
        
        if heading_matches:
            sections = []
            for i, match in enumerate(heading_matches):
                start = match.start()
                end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
                sections.append(text[start:end])
            return sections
        
        # Try function-based
        function_matches = list(self.function_pattern.finditer(text))
        if function_matches:
            sections = []
            for i, match in enumerate(function_matches):
                start = match.start()
                end = function_matches[i + 1].start() if i + 1 < len(function_matches) else len(text)
                sections.append(text[start:end])
            return sections
        
        # Fallback to paragraph-based
        paragraphs = re.split(r'\n\s*\n', text)
        return [p for p in paragraphs if p.strip()]

