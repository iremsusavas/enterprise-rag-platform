"""
Configuration file for Enterprise RAG Platform

This version is designed to work with open-source local models:
- sentence-transformers for embeddings
- Hugging Face transformers for the main LLM
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Embedding Models Configuration (sentence-transformers)
# All chosen models have 768-dimensional embeddings.
EMBEDDING_MODELS = {
    "policy": "sentence-transformers/all-mpnet-base-v2",         # General, strong semantic model
    "legal": "sentence-transformers/multi-qa-mpnet-base-dot-v1", # QA-optimized, good for legal Q&A
    "technical": "sentence-transformers/all-mpnet-base-v2",      # Good for technical/API text
}

# Chunking Configuration
CHUNK_CONFIG = {
    "policy": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "strategy": "section_based"
    },
    "legal": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "strategy": "semantic"
    },
    "technical": {
        "chunk_size": 800,
        "chunk_overlap": 100,
        "strategy": "function_based"
    }
}

# LLM Configuration (Hugging Face model id)
# Example strong open-source instruct models:
# - "meta-llama/Meta-Llama-3-8B-Instruct"
# - "mistralai/Mistral-7B-Instruct-v0.3"
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
LLM_TEMPERATURE = 0.0  # Deterministic for RAG-style usage

# Vector DB Configuration
VECTOR_DB_TYPE = "faiss"  # Options: "faiss", "weaviate", "pinecone"
VECTOR_DIMENSION = 768    # sentence-transformers mpnet-based models

# Evaluation Configuration
EVALUATION_METRICS = ["faithfulness", "completeness", "hallucination"]
EVALUATION_THRESHOLD = 3.0  # Minimum score (1-5 scale)

# Paths
DATA_DIR = "data"
INDICES_DIR = "indices"
LOGS_DIR = "logs"

