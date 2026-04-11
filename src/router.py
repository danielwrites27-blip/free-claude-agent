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
    # Sambanova reasoning model
    "DeepSeek-R1-0528",
]

# Provider name constants
GROQ = "groq"
SAMBANOVA = "sambanova"
CEREBRAS = "cerebras"

# Models that support native function/tool calling
# DeepSeek-R1 on Sambanova does NOT support tool calling yet
TOOL_CAPABLE_MODELS = {
    "llama-3.3-70b-versatile",       # Groq — primary tool calling model
    "llama-3.1-8b-instant",          # Groq — fallback tool calling model
    "llama-3.3-70b",                 # Cerebras
    "llama3.1-8b",                   # Cerebras
    "Meta-Llama-3.3-70B-Instruct",   # Sambanova (Llama supports it)
    "Meta-Llama-3.1-8B-Instruct",    # Sambanova (Llama supports it)
}


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

    def supports_tool_calling(self, model: str) -> bool:
        """Return True if this model supports native function/tool calling."""
        return model in TOOL_CAPABLE_MODELS

    def estimate_complexity(self, prompt: str) -> Literal["simple", "reasoning", "complex"]:
        """Classify query complexity for model selection"""
        words = prompt.split()
        word_count = len(words)
        lower = prompt.lower()

        has_reasoning_kw = any(kw in lower for kw in self.reasoning_keywords)
        has_code = bool(self.code_patterns.search(prompt))
        is_long = word_count > 150 or len(prompt) > 800
        multi_question = prompt.count("?") > 1

        if word_count < self.simple_threshold and not has_reasoning_kw and not has_code:
            if not multi_question:
                return "simple"

        if (is_long and has_reasoning_kw) or (has_code and has_reasoning_kw):
            return "complex"

        return "reasoning"

    def select_model(
        self,
        prompt: str,
        available_models: Dict[str, dict],
        force_reasoning: bool = False,
        require_tool_calling: bool = False,
    ) -> Tuple[str, str]:
        """
        Select (model_name, provider) based on complexity + availability.

        If require_tool_calling=True, only returns tool-capable models.
        This is used in normal mode to ensure function calling works.

        Priority philosophy:
        - Simple → 8B on Cerebras (fastest) → 8B on Groq → 8B on Sambanova
        - Reasoning → DeepSeek-R1 → 70B on Groq → 70B Sambanova → 70B Cerebras
        - Complex → same as reasoning
        - Tool calling required → skip DeepSeek-R1, prefer Groq 70B
        """
        complexity = "reasoning" if force_reasoning else self.estimate_complexity(prompt)

        priority_map = {
            "simple": [
                ("llama3.1-8b",                  CEREBRAS),
                ("llama-3.1-8b-instant",          GROQ),
                ("Meta-Llama-3.1-8B-Instruct",    SAMBANOVA),
            ],
            "reasoning": [
                ("DeepSeek-R1-0528",              SAMBANOVA),
                ("llama-3.3-70b-versatile",       GROQ),
                ("Meta-Llama-3.3-70B-Instruct",   SAMBANOVA),
                ("llama-3.3-70b",                 CEREBRAS),
                ("llama3.1-8b",                   CEREBRAS),
                ("llama-3.1-8b-instant",          GROQ),
            ],
            "complex": [
                ("DeepSeek-R1-0528",              SAMBANOVA),
                ("llama-3.3-70b-versatile",       GROQ),
                ("Meta-Llama-3.3-70B-Instruct",   SAMBANOVA),
                ("llama-3.3-70b",                 CEREBRAS),
                ("llama3.1-8b",                   CEREBRAS),
                ("llama-3.1-8b-instant",          GROQ),
            ],
        }

        for model, provider in priority_map.get(complexity, []):
            if model not in available_models and model not in available_models:
                continue
            if model not in available_models:
                continue
            # If tool calling is required, skip non-capable models
            if require_tool_calling and not self.supports_tool_calling(model):
                continue
            return model, provider

        # Ultimate fallback — always tool capable
        return "llama-3.1-8b-instant", GROQ

    def select_tool_capable_model(
        self,
        prompt: str,
        available_models: Dict[str, dict],
    ) -> Tuple[str, str]:
        """
        Shortcut: always returns a tool-capable model.
        Used in normal mode where tool calling must work.
        Prefers Groq 70B → Groq 8B as they have best tool calling support.
        """
        tool_priority = [
            ("llama-3.1-8b-instant",          GROQ),        # 70B broken on Groq for tool calling
            ("llama3.1-8b",                   CEREBRAS),
            ("Meta-Llama-3.3-70B-Instruct",   SAMBANOVA),
            ("llama-3.3-70b-versatile",       GROQ),
            ("llama-3.3-70b",                 CEREBRAS),
            ("Meta-Llama-3.1-8B-Instruct",    SAMBANOVA),
        ]
        for model, provider in tool_priority:
            if model in available_models:
                return model, provider
        return "llama-3.1-8b-instant", GROQ

    def get_complexity_label(self, prompt: str) -> str:
        """Human-readable complexity for UI display"""
        c = self.estimate_complexity(prompt)
        return {
            "simple":    "⚡ 8B",
            "reasoning": "🧠 DeepSeek-R1",
            "complex":   "🧠 DeepSeek-R1",
        }[c]
