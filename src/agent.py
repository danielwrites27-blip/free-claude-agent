"""
Token-optimized, free-tier AI agent.
Uses Groq API + caveman compression + SQLite memory.
"""
import os
import re
from datetime import datetime, timedelta
from typing import Optional

from groq import Groq
import tiktoken

from .caveman import CAVEMAN_SYSTEM_PROMPT, compress_response
from .memory import TokenEfficientMemory
from .router import ModelRouter

class FreeAgent:
    """100% free, token-optimized reasoning agent"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        daily_token_limit: int = 50000,
        memory_path: str = "agent_memory.mv2"
    ):
        # API client (Groq free tier)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY required. Get free key: https://console.groq.com/keys")
        
        self.client = Groq(api_key=self.api_key)
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # Budget tracking
        self.daily_token_limit = daily_token_limit
        self.tokens_used_today = 0
        self.last_reset = datetime.now()
        
        # Components
        self.memory = TokenEfficientMemory(memory_path)
        self.router = ModelRouter()
        
        # Available models (Groq free tier)
        self.available_models = {
            "llama-3.1-8b-instant": {"rpdlimit": 14400, "cost": "free"},
            "llama-3.3-70b-versatile": {"rpdlimit": 1000, "cost": "free"},
        }
    
    def _reset_daily_if_needed(self):
        """Reset token counter if 24h passed"""
        now = datetime.now()
        if now - self.last_reset > timedelta(hours=24):
            self.tokens_used_today = 0
            self.last_reset = now
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens for budget tracking"""
        return len(self.encoder.encode(text))
    
    def _inject_memory(self, prompt: str, max_mem_tokens: int = 1000) -> str:
        """Add relevant memory without exceeding token budget"""
        recalled = self.memory.recall(prompt, max_tokens=max_mem_tokens)
        if not recalled:
            return prompt
        return f"Relevant history:\n{recalled}\n\nCurrent task:\n{prompt}"
    
    def ask(self, prompt: str, max_output_tokens: int = 1024) -> str:
        """Main entry point: Ask the agent a question"""
        # Reset daily counter if needed
        self._reset_daily_if_needed()
        
        # Budget check
        if self.tokens_used_today >= self.daily_token_limit:
            return "⚠️ Free tier limit reached. Reset in 24h."
        
        # Enrich with memory
        enriched_prompt = self._inject_memory(prompt)
        
        # Route to cheapest capable model
        model = self.router.select_model(enriched_prompt, self.available_models)
        
        # Generate response with caveman system prompt
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CAVEMAN_SYSTEM_PROMPT},
                    {"role": "user", "content": enriched_prompt}
                ],
                max_tokens=max_output_tokens,
                temperature=0.1,  # Lower = more deterministic
                top_p=0.95
            )
            
            # Extract and compress output
            raw_output = response.choices[0].message.content
            compressed_output = compress_response(raw_output)
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            self.tokens_used_today += tokens_used
            
            # Store interaction in memory (non-blocking)
            self.memory.store(
    content=f"Q: {prompt}\nA: {compressed_output}",
    tags=["conversation"]
            )
            
            return compressed_output
            
        except Exception as e:
            # Fallback error message (minimal tokens)
            return f"Error: {str(e)[:100]}..."
    
    def get_usage(self) -> dict:
        """Get current token usage stats"""
        self._reset_daily_if_needed()
        return {
            "tokens_used_today": self.tokens_used_today,
            "daily_limit": self.daily_token_limit,
            "remaining": max(0, self.daily_token_limit - self.tokens_used_today),
            "reset_in_hours": max(0, 24 - (datetime.now() - self.last_reset).seconds // 3600)
        }
