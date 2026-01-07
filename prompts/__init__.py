"""
RAG Prompt Templates
Strict prompts to prevent hallucination
"""

from .rag_prompts import get_rag_prompt, get_evaluation_prompt

__all__ = ["get_rag_prompt", "get_evaluation_prompt"]

