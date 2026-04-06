"""
Smart model routing: Cheapest capable model for each query.
"""
import re
from typing import Literal

ModelName = Literal[
    "llama-3.1-8b-instant",      # Groq: Fast, 14,400 RPD free
    "llama-3.3-70b-versatile",   # Groq: Smart, 1,000 RPD free
    "qwen/qwen-2.5-72b-instruct" # OpenRouter fallback
]

class ModelRouter:
    """Route queries to cheapest capable model"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.simple_threshold = self.config.get("simple_token_threshold", 200)
        self.reasoning_keywords = set(
            self.config.get("reasoning_keywords", 
            ["why", "how", "analyze", "compare", "explain", "debug", "optimize", "fix"])
        )
    
    def estimate_complexity(self, prompt: str) -> Literal["simple", "reasoning", "complex"]:
        """Classify query complexity for model selection"""
        # Simple: Short, factual, no reasoning keywords
        if len(prompt.split()) < self.simple_threshold:
            if not any(kw in prompt.lower() for kw in self.reasoning_keywords):
                if "?" not in prompt or prompt.count("?") == 1:
                    return "simple"
        
        # Reasoning: Multi-step, explanatory, or technical
        if any(kw in prompt.lower() for kw in self.reasoning_keywords):
            return "reasoning"
        
        if len(prompt) > 500 or prompt.count("?") > 1:
            return "reasoning"
        
        # Complex: Fallback for edge cases
        return "complex"
    
    def select_model(self, prompt: str, available_models: dict) -> ModelName:
        """Select model based on complexity + availability"""
        complexity = self.estimate_complexity(prompt)
        
        # Priority order by cost (cheapest first)
        priority = {
            "simple": ["llama-3.1-8b-instant"],
            "reasoning": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            "complex": ["qwen/qwen-2.5-72b-instruct", "llama-3.3-70b-versatile"]
        }
        
        for model in priority.get(complexity, []):
            if model in available_models:
                return model
        
        # Ultimate fallback
        return "llama-3.1-8b-instant"
