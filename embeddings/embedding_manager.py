"""
Embedding Manager for multi-embedding strategy
Each index uses its own sentence-transformers model.
"""
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

import config


class EmbeddingManager:
    """
    Manages embeddings for different document types using sentence-transformers.

    Key principle: Query is embedded using the selected index's embedding model.
    This ensures semantic coordinate system alignment.
    """

    def __init__(self):
        """Initialize embedding models for each document type."""
        self.model_names = config.EMBEDDING_MODELS
        self.models = {
            doc_type: SentenceTransformer(model_name)
            for doc_type, model_name in self.model_names.items()
        }

    def _get_model(self, doc_type: str) -> SentenceTransformer:
        """Return the sentence-transformers model for a document type."""
        return self.models.get(doc_type, self.models["policy"])

    def embed_text(self, text: str, doc_type: str = "policy") -> np.ndarray:
        """
        Embed text using the appropriate model for document type.

        Args:
            text: Text to embed
            doc_type: Document type (policy, legal, technical)

        Returns:
            Embedding vector as numpy array
        """
        model = self._get_model(doc_type)
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding

    def embed_batch(
        self,
        texts: List[str],
        doc_type: str = "policy",
        batch_size: int = 100,
    ) -> List[np.ndarray]:
        """
        Embed multiple texts in batches.

        Args:
            texts: List of texts to embed
            doc_type: Document type
            batch_size: Number of texts to embed at once

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        model = self._get_model(doc_type)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return list(embeddings)

    def get_embedding_model(self, doc_type: str) -> str:
        """Get the embedding model name for a document type."""
        return self.model_names.get(doc_type, self.model_names["policy"])

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        # Assume all models share the same dimension as configured
        return config.VECTOR_DIMENSION

