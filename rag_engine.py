"""
Main RAG Engine
Orchestrates the entire RAG pipeline
"""
from typing import Dict, List

import config
from agents.query_router import QueryRouter
from embeddings.embedding_manager import EmbeddingManager
from evaluation.llm_judge import LLMJudge
from llm.llm_client import LLMClient
from prompts.rag_prompts import get_rag_prompt
from vector_db.vector_store_factory import VectorStoreFactory


class RAGEngine:
    """
    Main RAG Engine that orchestrates the entire pipeline:
    1. Query Routing
    2. Embedding
    3. Retrieval
    4. Generation
    5. Evaluation
    """
    
    def __init__(self):
        """
        Initialize RAG Engine (no external API key required).
        """
        self.router = QueryRouter()
        self.embedding_manager = EmbeddingManager()
        self.judge = LLMJudge()
        self.llm = LLMClient()
        
        # Initialize vector stores for each index
        self.vector_stores = {}
        for index_name in ["policy", "legal", "technical"]:
            self.vector_stores[index_name] = VectorStoreFactory.create_store(index_name)
            # Try to load existing index
            try:
                self.vector_stores[index_name].load()
            except FileNotFoundError:
                pass  # Index doesn't exist yet, will be created on ingestion
    
    def query(self, user_query: str, k: int = 5, evaluate: bool = True) -> Dict:
        """
        Process a user query through the RAG pipeline
        
        Args:
            user_query: User's question
            k: Number of chunks to retrieve
            evaluate: Whether to run evaluation
            
        Returns:
            Dictionary with answer, sources, routing info, and evaluation
        """
        # Step 1: Route query
        routing_result = self.router.route_query(user_query)
        selected_index = routing_result["selected_index"]
        
        # Step 2: Embed query using selected index's embedding model
        query_embedding = self.embedding_manager.embed_text(user_query, selected_index)
        
        # Step 3: Retrieve relevant chunks
        vector_store = self.vector_stores[selected_index]
        retrieved_chunks = vector_store.search(query_embedding, k=k)
        
        # Step 4: Format context
        context_parts = []
        source_chunks = []
        for i, result in enumerate(retrieved_chunks):
            chunk_content = result["chunk"]
            metadata = result["metadata"]
            context_parts.append(f"Chunk {i+1}:\n{chunk_content}")
            source_chunks.append({
                "chunk_id": metadata.get("chunk_id", f"chunk_{i+1}"),
                "source": metadata.get("file_name", "unknown"),
                "score": result["score"]
            })
        
        context = "\n\n".join(context_parts)
        
        # Step 5: Generate answer with strict prompt
        prompt = get_rag_prompt(context, user_query)

        try:
            answer = self.llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based ONLY on the provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_new_tokens=512,
                temperature=config.LLM_TEMPERATURE,
            )
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        # Step 6: Evaluate (optional)
        evaluation = None
        if evaluate:
            evaluation = self.judge.evaluate(user_query, answer, context)
        
        return {
            "answer": answer,
            "sources": source_chunks,
            "routing": routing_result,
            "evaluation": evaluation,
            "context_used": len(retrieved_chunks)
        }
    
    def ingest_documents(self, documents: List[Dict], doc_type: str):
        """
        Ingest documents into the system
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            doc_type: Type of documents (policy, legal, technical)
        """
        from chunking.chunking_factory import ChunkingFactory
        
        # Get appropriate chunker
        chunker = ChunkingFactory.get_chunker(doc_type)
        
        # Chunk all documents
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk(doc["content"], doc["metadata"])
            all_chunks.extend(chunks)
        
        # Generate embeddings
        chunk_texts = [chunk["content"] for chunk in all_chunks]
        embeddings = self.embedding_manager.embed_batch(chunk_texts, doc_type)
        
        # Prepare metadata with chunk IDs
        metadatas = []
        for i, chunk in enumerate(all_chunks):
            chunk_id = f"{doc_type}_{i}_{hash(chunk['content']) % 10000}"
            metadata = {
                **chunk["metadata"],
                "chunk_id": chunk_id,
                "content": chunk["content"]
            }
            metadatas.append(metadata)
        
        # Add to vector store
        vector_store = self.vector_stores[doc_type]
        vector_store.add_vectors(embeddings, metadatas)
        vector_store.save()
        
        return len(all_chunks)

