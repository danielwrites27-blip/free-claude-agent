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
        
        # Check for URL in prompt and fetch content
        url_context = ""
        if "http://" in prompt or "https://" in prompt:
            urls = re.findall(r'https?://[^\s<>"]+', prompt)
            if urls:
                url = urls[0]
                fetched_content = self._fetch_url_content(url)
                if not fetched_content.startswith("Error"):
                    url_context = f"\n\nContent from {url}:\n{fetched_content}"
        
        memory_text = recalled if recalled else ""
        full_context = memory_text + url_context
        
        if not full_context:
            return prompt
        return f"Relevant context:\n{full_context}\n\nCurrent task:\n{prompt}"

    def _fetch_url_content(self, url: str) -> str:
        """Fetch and clean text content from a URL."""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, nav, footer elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            
            # Truncate to avoid token overflow (max 2000 chars)
            return text[:2000] if len(text) > 2000 else text
            
        except Exception as e:
            return f"Error fetching URL: {str(e)}"
    
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
                tags=["conversation"],
                token_count=tokens_used
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
