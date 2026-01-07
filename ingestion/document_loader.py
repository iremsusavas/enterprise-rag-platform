"""
Document Loader for various file formats
"""
import os
from typing import List, Dict, Optional
from pathlib import Path
import pypdf
from docx import Document
import markdown
from bs4 import BeautifulSoup


class DocumentLoader:
    """Loads documents from various formats and extracts text"""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.txt': self._load_txt,
            '.md': self._load_markdown,
            '.html': self._load_html
        }
    
    def load_document(self, file_path: str, doc_type: str = "policy") -> Dict:
        """
        Load a document and return its content with metadata
        
        Args:
            file_path: Path to the document
            doc_type: Type of document (policy, legal, technical)
            
        Returns:
            Dictionary with 'content' and 'metadata'
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}")
        
        loader_func = self.supported_formats[extension]
        content = loader_func(file_path)
        
        metadata = {
            "file_path": str(file_path),
            "file_name": path.name,
            "doc_type": doc_type,
            "file_size": path.stat().st_size,
            "extension": extension
        }
        
        return {
            "content": content,
            "metadata": metadata
        }
    
    def load_directory(self, directory_path: str, doc_type: str = "policy") -> List[Dict]:
        """
        Load all supported documents from a directory
        
        Args:
            directory_path: Path to directory
            doc_type: Type of documents in directory
            
        Returns:
            List of document dictionaries
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        documents = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    doc = self.load_document(str(file_path), doc_type)
                    documents.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _load_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        text_parts = []
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())
        return "\n\n".join(text_parts)
    
    def _load_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n\n".join(paragraphs)
    
    def _load_txt(self, file_path: str) -> str:
        """Load plain text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _load_markdown(self, file_path: str) -> str:
        """Load markdown file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _load_html(self, file_path: str) -> str:
        """Extract text from HTML"""
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            return soup.get_text()

