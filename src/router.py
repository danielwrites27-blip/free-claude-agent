"""
Smart model routing: Cheapest capable model for each query.
Supports Groq, Sambanova, Cerebras — all free tier.
"""
import re
from typing import Literal, Dict, Tuple

# All supported model names across providers
ModelName = Literal[
    # Groq (fast inference, strict RPD limits)
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    # Sambanova (generous free tier, good for long context)
    "Meta-Llama-3.1-8B-Instruct",
    "Meta-Llama-3.3-70B-Instruct",
    # Cerebras (fastest inference, wafer-scale)
    "llama3.1-8b",
    "llama-3.3-70b",
]

# Provider name constants
GROQ = "groq"
SAMBANOVA = "sambanova"
CEREBRAS = "cerebras"


class ModelRouter:
    """Route queries to cheapest capable model across providers"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.simple_threshold = self.config.get("simple_token_threshold", 100)

        self.reasoning_keywords = set(self.config.get("reasoning_keywords", [
            "why", "how", "analyze", "compare", "explain", "debug",
            "what does", "what is", "what are", "how does", "tell me",
            "who is", "who are", "search for", "look up", "latest", "current",
            "optimize", "fix", "difference", "tradeoff", "architecture",
            "design", "implement", "write", "create", "build", "review",
            "refactor", "plan", "strategy", "best", "recommend"
        ]))

        self.code_patterns = re.compile(
            r'(def |class |import |```|function |->|=>|SELECT |FROM |\$\{)',
            re.IGNORECASE
        )

    def estimate_complexity(self, prompt: str) -> Literal["simple", "reasoning", "complex"]:
        """Classify query complexity for model selection"""
        words = prompt.split()
        word_count = len(words)
        lower = prompt.lower()

        has_reasoning_kw = any(kw in lower for kw in self.reasoning_keywords)
        has_code = bool(self.code_patterns.search(prompt))
        is_long = word_count > 150 or len(prompt) > 800
        multi_question = prompt.count("?") > 1

        # Simple: short, no reasoning keywords, no code, single question
        if word_count < self.simple_threshold and not has_reasoning_kw and not has_code:
            if not multi_question:
                return "simple"

        # Complex: long OR has code AND reasoning — needs best model
        if (is_long and has_reasoning_kw) or (has_code and has_reasoning_kw):
            return "complex"

        # Reasoning: everything in between
        return "reasoning"

    def select_model(
        self,
        prompt: str,
        available_models: Dict[str, dict],
        force_reasoning: bool = False
    ) -> Tuple[str, str]:
        """
        Select (model_name, provider) based on complexity + availability.
        Returns a tuple so agent knows which client to use.
        
        Priority philosophy:
        - Simple → 8B on Cerebras (fastest) → 8B on Groq → 8B on Sambanova
        - Reasoning → 70B on Groq → 70B on Sambanova → 70B on Cerebras → fall back to 8B
        - Complex → same as reasoning (no 405B on free tier)
        """
        complexity = "reasoning" if force_reasoning else self.estimate_complexity(prompt)

        # Priority lists: (model_name, provider)
        priority_map = {
            "simple": [
                ("llama3.1-8b", CEREBRAS),
                ("llama-3.1-8b-instant", GROQ),
                ("Meta-Llama-3.1-8B-Instruct", SAMBANOVA),
            ],
            "reasoning": [
                ("llama-3.3-70b-versatile", GROQ),
                ("Meta-Llama-3.3-70B-Instruct", SAMBANOVA),
                ("llama-3.3-70b", CEREBRAS),
                # Fallback to 8B if no 70B available
                ("llama3.1-8b", CEREBRAS),
                ("llama-3.1-8b-instant", GROQ),
            ],
            "complex": [
                ("llama-3.3-70b-versatile", GROQ),
                ("Meta-Llama-3.3-70B-Instruct", SAMBANOVA),
                ("llama-3.3-70b", CEREBRAS),
                ("llama3.1-8b", CEREBRAS),
                ("llama-3.1-8b-instant", GROQ),
            ],
        }

        for model, provider in priority_map.get(complexity, []):
            key = f"{provider}:{model}"
            # Check if this provider+model combo is available
            if key in available_models or model in available_models:
                return model, provider

        # Ultimate fallback — should never hit this
        return "llama-3.1-8b-instant", GROQ

    def get_complexity_label(self, prompt: str) -> str:
        """Human-readable complexity for UI display"""
        c = self.estimate_complexity(prompt)
        return {"simple": "⚡ 8B", "reasoning": "🧠 70B", "complex": "🧠 70B"}[c]
