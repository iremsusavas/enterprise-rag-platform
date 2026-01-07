"""
FAISS Vector Store Implementation
Local vector database for embeddings
"""
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Optional, Tuple
import config


class FAISSStore:
    """
    FAISS-based vector store
    
    Stores vectors with metadata for retrieval
    """
    
    def __init__(self, index_name: str, dimension: int = None):
        """
        Initialize FAISS store
        
        Args:
            index_name: Name of the index (e.g., "policy", "legal", "technical")
            dimension: Dimension of vectors (defaults to config)
        """
        self.index_name = index_name
        self.dimension = dimension or config.VECTOR_DIMENSION
        self.index = None
        self.metadata_store = []  # List of metadata dicts, aligned with index
        self.id_to_index = {}  # Map chunk_id to index position
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        # Using L2 distance (Euclidean) - can switch to cosine similarity
        self.index = faiss.IndexFlatL2(self.dimension)
    
    def add_vectors(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        """
        Add vectors to the index
        
        Args:
            vectors: List of embedding vectors
            metadatas: List of metadata dictionaries (one per vector)
        """
        if len(vectors) != len(metadatas):
            raise ValueError("Vectors and metadatas must have same length")
        
        # Convert to numpy array
        vectors_array = np.array(vectors).astype('float32')
        
        # Add to FAISS index
        self.index.add(vectors_array)
        
        # Store metadata
        start_idx = len(self.metadata_store)
        for i, metadata in enumerate(metadatas):
            chunk_id = metadata.get("chunk_id", f"{self.index_name}_{start_idx + i}")
            self.metadata_store.append(metadata)
            self.id_to_index[chunk_id] = start_idx + i
    
    def search(self, query_vector: np.ndarray, k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"doc_type": "policy"})
            
        Returns:
            List of dictionaries with 'chunk', 'metadata', and 'score'
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure query vector is correct shape
        query_vector = query_vector.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_vector, min(k * 2, self.index.ntotal))
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            metadata = self.metadata_store[idx]
            
            # Apply metadata filters if provided
            if filter_metadata:
                if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            
            results.append({
                "chunk": metadata.get("content", ""),
                "metadata": metadata,
                "score": float(distance)  # L2 distance (lower is better)
            })
            
            if len(results) >= k:
                break
        
        return results
    
    def save(self, directory: str = None):
        """
        Save index to disk
        
        Args:
            directory: Directory to save to (defaults to indices_dir from config)
        """
        if directory is None:
            directory = config.INDICES_DIR
        
        os.makedirs(directory, exist_ok=True)
        
        index_path = os.path.join(directory, f"{self.index_name}.index")
        metadata_path = os.path.join(directory, f"{self.index_name}_metadata.pkl")
        
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                "metadata_store": self.metadata_store,
                "id_to_index": self.id_to_index
            }, f)
    
    def load(self, directory: str = None):
        """
        Load index from disk
        
        Args:
            directory: Directory to load from (defaults to indices_dir from config)
        """
        if directory is None:
            directory = config.INDICES_DIR
        
        index_path = os.path.join(directory, f"{self.index_name}.index")
        metadata_path = os.path.join(directory, f"{self.index_name}_metadata.pkl")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata_store = data["metadata_store"]
            self.id_to_index = data["id_to_index"]
    
    def get_stats(self) -> Dict:
        """Get statistics about the index"""
        return {
            "index_name": self.index_name,
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "metadata_count": len(self.metadata_store)
        }

