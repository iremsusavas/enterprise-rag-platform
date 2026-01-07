"""
Intelligent Query Router Agent
Uses an LLM to route queries to appropriate indexes
"""
import json
from typing import Dict

import config
from llm.llm_client import LLMClient


class QueryRouter:
    """
    Routes user queries to appropriate document indexes
    
    Key insight: Uses LLM not to generate answers, but to make routing decisions
    """
    
    def __init__(self):
        """
        Initialize query router
        
        Args:
            (no external API key required; uses local/open-source model)
        """
        self.llm = LLMClient()
        self.available_indexes = ["policy", "legal", "technical"]
    
    def route_query(self, query: str) -> Dict:
        """
        Route query to appropriate index
        
        Args:
            query: User query
            
        Returns:
            Dictionary with 'selected_index', 'reason', and 'confidence'
        """
        prompt = self._create_routing_prompt(query)
        
        try:
            raw = self.llm.generate(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                max_new_tokens=256,
                temperature=0.0,
            )
            result = self._parse_json_response(raw)
            
            # Validate result
            if "selected_index" not in result:
                raise ValueError("Router did not return selected_index")
            
            if result["selected_index"] not in self.available_indexes:
                # Fallback to policy if invalid
                result["selected_index"] = "policy"
                result["reason"] = "Invalid index selected, defaulting to policy"
            
            return result
            
        except Exception as e:
            # Fallback to policy index on error
            return {
                "selected_index": "policy",
                "reason": f"Routing error: {str(e)}, defaulting to policy",
                "confidence": 0.5
            }

    def _parse_json_response(self, text: str) -> Dict:
        """Best-effort extraction of JSON object from model output."""
        try:
            return json.loads(text)
        except Exception:
            # Try to find the first {...} block
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass
        # Fallback minimal structure
        return {
            "selected_index": "policy",
            "reason": "Failed to parse JSON from router LLM output.",
            "confidence": 0.5,
        }
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for routing"""
        return """You are an intelligent query router for an enterprise RAG system.

Your job is to analyze user queries and determine which document index is most appropriate:
- policy: Employee policies, HR guidelines, company rules, procedures
- legal: Contracts, legal documents, compliance, regulatory requirements
- technical: API documentation, technical specifications, code documentation, engineering docs

Return a JSON object with:
- selected_index: one of ["policy", "legal", "technical"]
- reason: brief explanation of your choice
- confidence: float between 0 and 1

Be decisive and choose the most appropriate index."""
    
    def _create_routing_prompt(self, query: str) -> str:
        """Create routing prompt for the query"""
        return f"""Analyze this user query and determine which document index should be used:

Query: "{query}"

Consider:
- Policy queries: company policies, employee handbooks, HR procedures, workplace rules
- Legal queries: contracts, legal terms, compliance, regulatory requirements, obligations
- Technical queries: API documentation, technical specs, code examples, engineering documentation

Return JSON with selected_index, reason, and confidence."""

