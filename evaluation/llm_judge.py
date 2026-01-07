"""
LLM-Judge Evaluation Pipeline
Automatically evaluates RAG responses for quality
"""
import json
from typing import Dict

import config
from llm.llm_client import LLMClient
from prompts.rag_prompts import get_evaluation_prompt


class LLMJudge:
    """
    LLM-based judge for evaluating RAG responses
    
    Evaluates:
    - Faithfulness: Is answer faithful to context?
    - Completeness: Does answer fully address query?
    - Hallucination: Does answer contain fabricated info?
    """
    
    def __init__(self):
        """
        Initialize LLM Judge (uses local/open-source model).
        """
        self.llm = LLMClient()
    
    def evaluate(self, query: str, answer: str, context: str) -> Dict:
        """
        Evaluate a RAG response
        
        Args:
            query: Original user query
            answer: Generated answer
            context: Retrieved context chunks
            
        Returns:
            Dictionary with evaluation scores and reasoning
        """
        prompt = get_evaluation_prompt(query, answer, context)

        try:
            raw = self.llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator of RAG systems. Be strict and objective.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_new_tokens=256,
                temperature=0.0,  # Deterministic evaluation
            )

            result = self._parse_json_response(raw)
            
            # Validate and normalize scores
            metrics = ["faithfulness", "completeness", "hallucination"]
            for metric in metrics:
                if metric not in result:
                    result[metric] = 3.0  # Default score
                else:
                    result[metric] = float(result[metric])
                    # Clamp to 1-5 range
                    result[metric] = max(1.0, min(5.0, result[metric]))
            
            # Calculate overall if not present
            if "overall_score" not in result:
                result["overall_score"] = sum(result[metric] for metric in metrics) / len(metrics)
            else:
                result["overall_score"] = float(result["overall_score"])
            
            return result
            
        except Exception as e:
            # Return default scores on error
            return {
                "faithfulness": 3.0,
                "completeness": 3.0,
                "hallucination": 3.0,
                "overall_score": 3.0,
                "reasoning": f"Evaluation error: {str(e)}"
            }

    def _parse_json_response(self, text: str) -> Dict:
        """Best-effort extraction of JSON object from model output."""
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass
        return {
            "faithfulness": 3.0,
            "completeness": 3.0,
            "hallucination": 3.0,
            "overall_score": 3.0,
            "reasoning": "Failed to parse JSON from judge LLM output.",
        }
    
    def is_acceptable(self, evaluation: Dict, threshold: float = None) -> bool:
        """
        Check if evaluation meets quality threshold
        
        Args:
            evaluation: Evaluation result dictionary
            threshold: Minimum acceptable score (defaults to config)
            
        Returns:
            True if acceptable, False otherwise
        """
        threshold = threshold or config.EVALUATION_THRESHOLD
        overall = evaluation.get("overall_score", 0.0)
        return overall >= threshold

