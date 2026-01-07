"""
Strict RAG Prompts
Designed to prevent hallucination and ensure faithfulness to context
"""


def get_rag_prompt(context: str, query: str) -> str:
    """
    Get strict RAG prompt that prevents hallucination
    
    Args:
        context: Retrieved context chunks
        query: User query
        
    Returns:
        Formatted prompt
    """
    return f"""You are a retrieval-augmented assistant. Your role is to answer questions based ONLY on the provided context.

CRITICAL RULES:
1. Answer ONLY using information from the provided context below
2. If the answer is not in the context, explicitly say "I don't know" or "The information is not available in the provided documents"
3. Do NOT make up information, even if you think you know the answer
4. Do NOT use information from your training data that is not in the context
5. Cite which chunks you used in your answer

Context:
{context}

Question: {query}

Instructions:
- Provide a clear, accurate answer based solely on the context
- If you cannot answer from the context, say so explicitly
- List the source chunks you used (e.g., "Based on chunks 1, 3, and 5")
- Be precise and avoid speculation

Answer:"""


def get_evaluation_prompt(query: str, answer: str, context: str) -> str:
    """
    Get prompt for LLM-Judge evaluation
    
    Args:
        query: Original query
        answer: Generated answer
        context: Retrieved context
        
    Returns:
        Evaluation prompt
    """
    return f"""You are an LLM-Judge evaluating a RAG system's response.

Evaluate the following answer based on these criteria:

1. FAITHFULNESS (1-5): Is the answer faithful to the provided context? Does it contain information that contradicts or goes beyond the context?
   - 5: Completely faithful, all information from context
   - 3: Mostly faithful, minor additions
   - 1: Contains significant information not in context (hallucination)

2. COMPLETENESS (1-5): Does the answer fully address the query?
   - 5: Fully addresses all aspects of the query
   - 3: Addresses main points but misses some details
   - 1: Does not address the query adequately

3. HALLUCINATION (1-5): Does the answer contain fabricated information?
   - 5: No hallucination detected
   - 3: Minor unsupported claims
   - 1: Clear hallucination, information not in context

Query: {query}

Context:
{context}

Answer: {answer}

Provide your evaluation as JSON:
{{
    "faithfulness": <score 1-5>,
    "completeness": <score 1-5>,
    "hallucination": <score 1-5>,
    "overall_score": <average of three scores>,
    "reasoning": "<brief explanation>"
}}"""

