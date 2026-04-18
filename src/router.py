"""
Smart model routing: Cheapest capable model for each query.
Supports Groq, Sambanova, Cerebras, NVIDIA NIM — all free tier.
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
    # Sambanova reasoning model (base DeepSeek-R1 671B — R1-0528 deprecated Mar 31 2026)
    "DeepSeek-R1",
    # Cerebras (fastest inference, wafer-scale)
    "llama3.1-8b",
    "llama-3.3-70b",
    "qwen-3-235b-a22b-instruct-2507",
    # NVIDIA NIM (hybrid MoE, 3.5B active / 30B total, 1M context)
    "nvidia/nemotron-3-nano-30b-a3b",
    # Modal (GLM-5.1-FP8 — #1 SWE-Bench Pro, 754B MoE, 40B active, 200K context)
    "zai-org/GLM-5.1-FP8",
    # OpenRouter / Together AI (GLM-5.1 — permanent fallback after Modal)
    "z-ai/glm-5.1",
    # MiniMax M2.7 (via NVIDIA NIM — general reasoning)
    "minimaxai/minimax-m2.7",
]

# Provider name constants
GROQ = "groq"
SAMBANOVA = "sambanova"
CEREBRAS = "cerebras"
NVIDIA = "nvidia"
MODAL = "modal"
MINIMAX = "minimax"
OPENROUTER = "openrouter"
TOGETHER = "together"

# Models that support native function/tool calling
# DeepSeek-R1 on Sambanova does NOT support tool calling
TOOL_CAPABLE_MODELS = {
    "llama-3.3-70b-versatile",            # Groq — BROKEN for tool calling (XML bug)
    "llama-3.1-8b-instant",               # Groq — fallback tool calling model
    "llama3.1-8b",                        # Cerebras
    "qwen-3-235b-a22b-instruct-2507",     # Cerebras — verified working
    "Meta-Llama-3.3-70B-Instruct",        # Sambanova
    "Meta-Llama-3.1-8B-Instruct",         # Sambanova
    "nvidia/nemotron-3-nano-30b-a3b",     # NVIDIA NIM — S15 benchmark confirmed
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
        - Simple -> 8B on Cerebras (fastest) -> 8B on Groq -> 8B on Sambanova
        - Reasoning -> DeepSeek-R1 -> Nemotron-30B -> 70B Sambanova -> 70B Cerebras
        - Complex -> same as reasoning
        - Tool calling required -> skip DeepSeek-R1, use Qwen3 -> Nemotron -> Llama-70B
        """
        complexity = "reasoning" if force_reasoning else self.estimate_complexity(prompt)

        priority_map = {
            "simple": [
                ("llama3.1-8b",                   CEREBRAS),
                ("llama-3.1-8b-instant",           GROQ),
                ("Meta-Llama-3.1-8B-Instruct",     SAMBANOVA),
            ],
            "reasoning": [
                ("DeepSeek-R1",                    SAMBANOVA),
                ("llama-3.3-70b-versatile",        GROQ),
                ("nvidia/nemotron-3-nano-30b-a3b",  NVIDIA),
                ("Meta-Llama-3.3-70B-Instruct",    SAMBANOVA),
                ("llama-3.3-70b",                  CEREBRAS),
                ("llama3.1-8b",                    CEREBRAS),
                ("llama-3.1-8b-instant",           GROQ),
            ],
            "complex": [
                ("DeepSeek-R1",                    SAMBANOVA),
                ("llama-3.3-70b-versatile",        GROQ),
                ("nvidia/nemotron-3-nano-30b-a3b",  NVIDIA),
                ("Meta-Llama-3.3-70B-Instruct",    SAMBANOVA),
                ("llama-3.3-70b",                  CEREBRAS),
                ("llama3.1-8b",                    CEREBRAS),
                ("llama-3.1-8b-instant",           GROQ),
            ],
        }

        for model, provider in priority_map.get(complexity, []):
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

        Chain (S15 benchmark verified):
          Qwen3-235B (Cerebras)        — PRIMARY, best quality
          Nemotron-3-Nano-30B (NVIDIA) — Fallback1, fast + reliable tool calling
          Llama-3.3-70B (SambaNova)    — Fallback2
          Llama-3.1-8B (SambaNova)     — Last resort
          Llama-3.1-8B (Groq)          — Ultimate fallback (hardcoded below)

        Note: Groq 70B broken for tool calling (XML format bug — monitor for fix).
        """
        tool_priority = [
            ("qwen-3-235b-a22b-instruct-2507",  CEREBRAS),  # PRIMARY — best tool calling quality
            ("nvidia/nemotron-3-nano-30b-a3b",   NVIDIA),    # Fallback1 — S15: speed+tool+coding PASS
            ("Meta-Llama-3.3-70B-Instruct",      SAMBANOVA), # Fallback2
            ("Meta-Llama-3.1-8B-Instruct",       SAMBANOVA), # Last resort
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
