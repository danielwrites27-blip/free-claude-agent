"""
Free Claude Agent — Token-optimized, 100% free AI agent.

Usage:
    from src import FreeAgent
    agent = FreeAgent(api_key="your_groq_key")
    response = agent.ask("How do I fix a React re-render bug?")
"""

__version__ = "0.1.0"
__author__ = "Free Claude Agent Contributors"
__license__ = "MIT"

# Export main classes for easy imports
from .agent import FreeAgent
from .caveman import CAVEMAN_SYSTEM_PROMPT, compress_response
from .memory import TokenEfficientMemory
from .router import ModelRouter, ModelName

# Convenience: list available exports
__all__ = [
    "FreeAgent",
    "CAVEMAN_SYSTEM_PROMPT",
    "compress_response", 
    "TokenEfficientMemory",
    "ModelRouter",
    "ModelName",
    "__version__",
    "__author__",
    "__license__",
]
