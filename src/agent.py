"""
Token-optimized, free-tier AI agent.
Uses Groq (primary) → Cerebras → SambaNova fallback chain.
Upgraded: Chain-of-thought reasoning, working memory, metrics.
"""
import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from groq import Groq
from openai import OpenAI
import tiktoken

from .caveman import (
    CAVEMAN_SYSTEM_PROMPT,
    compress_response,
    REASONING_PROMPT_TEMPLATE,
    PLAN_PROMPT_TEMPLATE,
    TEST_FIRST_PROMPT,
    is_reasoning_query
)
from .memory import TokenEfficientMemory, WorkingMemory, ConversationSummarizer
from .router import ModelRouter


class AgentMetrics:
    """Track performance metrics for optimization"""

    def __init__(self):
        self.queries: List[Dict] = []

    def log(self, prompt: str, response: str, tokens_used: int,
            latency_ms: float, model: str, success: bool = True):
        self.queries.append({
            "prompt_len": len(prompt),
            "response_len": len(response),
            "tokens": tokens_used,
            "latency_ms": latency_ms,
            "model": model,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })

    def get_report(self) -> Dict:
        if not self.queries:
            return {}
        successful = [q for q in self.queries if q["success"]]
        return {
            "total_queries": len(self.queries),
            "successful_queries": len(successful),
            "success_rate": len(successful) / len(self.queries),
            "avg_tokens_per_query": sum(q["tokens"] for q in self.queries) / len(self.queries),
            "avg_latency_ms": sum(q["latency_ms"] for q in self.queries) / len(self.queries),
            "token_efficiency": sum(q["response_len"] for q in self.queries) /
                                max(1, sum(q["tokens"] for q in self.queries)),
            "models_used": list(set(q["model"] for q in self.queries))
        }

    def reset(self):
        self.queries = []


class FreeAgent:
    """100% free, token-optimized reasoning agent with provider fallback"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        daily_token_limit: int = 50000,
        memory_path: str = "agent_memory.db",
        enable_reasoning: bool = True,
        enable_code_execution: bool = False
    ):
        # Groq client (primary)
        self.groq_api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY required. Get free key: https://console.groq.com/keys")
        self.groq_client = Groq(api_key=self.groq_api_key)

        # Cerebras client (fallback 1)
        self.cerebras_key = os.getenv("CEREBRAS_API_KEY")
        self.cerebras_client = OpenAI(
            api_key=self.cerebras_key or "none",
            base_url="https://api.cerebras.ai/v1"
        ) if self.cerebras_key else None

        # SambaNova client (fallback 2)
        self.sambanova_key = os.getenv("SAMBANOVA_API_KEY")
        self.sambanova_client = OpenAI(
            api_key=self.sambanova_key or "none",
            base_url="https://api.sambanova.ai/v1"
        ) if self.sambanova_key else None

        self.encoder = tiktoken.get_encoding("cl100k_base")

        # Budget tracking
        self.daily_token_limit = daily_token_limit
        self.tokens_used_today = 0
        self.last_reset = datetime.now()

        # Feature flags
        self.enable_reasoning = enable_reasoning
        self.enable_code_execution = enable_code_execution

        # Components
        self.memory = TokenEfficientMemory(memory_path)
        self.router = ModelRouter()
        self.working_memory = WorkingMemory()
        self.summarizer = ConversationSummarizer()
        self.metrics = AgentMetrics()

        # Provider configs - ordered by preference
        self.providers = [
            {
                "name": "Groq",
                "client": self.groq_client,
                "model": "llama-3.3-70b-versatile",
                "type": "groq"
            },
            {
                "name": "Cerebras",
                "client": self.cerebras_client,
                "model": "gpt-oss-120b",
                "type": "openai"
            },
            {
                "name": "SambaNova",
                "client": self.sambanova_client,
                "model": "Llama-3.3-70B",
                "type": "openai"
            }
        ]

        self.conversation_history: List[Dict] = []

    def _reset_daily_if_needed(self):
        now = datetime.now()
        if now - self.last_reset > timedelta(hours=24):
            self.tokens_used_today = 0
            self.last_reset = now
            self.working_memory.clear()

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _inject_memory(self, prompt: str, max_mem_tokens: int = 1000) -> str:
        recalled = self.memory.recall(prompt, max_tokens=max_mem_tokens)
        if not recalled:
            return prompt
        return f"Relevant history:\n{recalled}\n\nCurrent task:\n{prompt}"

    def _call_provider(self, provider: dict, messages: list, max_tokens: int):
        """Call a provider, return (response_text, tokens_used)"""
        response = provider["client"].chat.completions.create(
            model=provider["model"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.95
        )
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else self._count_tokens(text)
        return text, tokens

    def _generate_raw(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate with provider fallback, no compression"""
        start_time = time.time()
        messages = [
            {"role": "system", "content": CAVEMAN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        last_error = None
        for provider in self.providers:
            if provider["client"] is None:
                continue
            try:
                text, tokens = self._call_provider(provider, messages, max_tokens)
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.log(prompt, text, tokens, latency_ms, provider["name"])
                return text
            except Exception as e:
                last_error = str(e)
                continue
        return f"⚠️ All providers failed. Last error: {last_error[:150]}"

    def _apply_reasoning_template(self, prompt: str) -> str:
        return REASONING_PROMPT_TEMPLATE.format(query=prompt)

    def _plan_and_execute(self, goal: str, max_steps: int = 5) -> str:
        plan_prompt = PLAN_PROMPT_TEMPLATE.format(goal=goal, max_steps=max_steps)
        plan_response = self._generate_raw(plan_prompt, max_tokens=256)

        steps = []
        for line in plan_response.split('\n'):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, max_steps + 1)):
                step_text = line.split('.', 1)[1].strip() if '.' in line else line
                steps.append(step_text)

        if not steps:
            steps = [goal]

        results = []
        for i, step in enumerate(steps[:max_steps], 1):
            step_prompt = f"Step {i}/{len(steps)}: {step}\n\nPrevious: {'; '.join(results[-2:]) if results else 'None'}\n\nExecute:"
            result = self._generate_raw(step_prompt, max_tokens=512)
            results.append(f"Step {i}: {result[:200]}")
            self.working_memory.add(f"Step {i}", result[:150], confidence=0.8)

        synthesis_prompt = f"Synthesize final answer from:\n{chr(10).join(results)}\n\nGoal: {goal}\n\nFinal Answer:"
        return self._generate_raw(synthesis_prompt, max_tokens=512)

    def _should_summarize_conversation(self) -> bool:
        total_tokens = sum(self._count_tokens(msg["content"])
                          for msg in self.conversation_history)
        return total_tokens > 3000

    def _summarize_and_trim_history(self):
        if len(self.conversation_history) < 10:
            return
        recent = self.conversation_history[-5:]
        to_summarize = self.conversation_history[:-5]
        if not to_summarize:
            return
        summary = self.summarizer.summarize(to_summarize, self._generate_raw)
        self.conversation_history = [
            {"role": "system", "content": f"Conversation summary:\n{summary}"},
            *recent
        ]

    def ask(self, prompt: str, max_output_tokens: int = 1024,
            use_reasoning: Optional[bool] = None,
            is_coding_task: bool = False) -> str:
        """Main entry point with provider fallback"""
        start_time = time.time()
        self._reset_daily_if_needed()

        if self.tokens_used_today >= self.daily_token_limit:
            return "⚠️ Daily token limit reached. Resets in 24h."

        if use_reasoning is None:
            use_reasoning = self.enable_reasoning and is_reasoning_query(prompt)

        if not is_coding_task:
            is_coding_task = any(kw in prompt.lower() for kw in
                                 ["code", "python", "function", "debug", "write code", "fix"])

        working_context = self.working_memory.get_context()
        memory_context = self._inject_memory(prompt)
        base_prompt = f"{working_context}\n\n{memory_context}\n\nUser: {prompt}".strip()

        if use_reasoning:
            base_prompt = self._apply_reasoning_template(prompt)

        try:
            if use_reasoning:
                raw_output = self._plan_and_execute(prompt)
            else:
                raw_output = self._generate_raw(base_prompt, max_tokens=max_output_tokens)

            compressed_output = compress_response(raw_output)
            tokens_used = self._count_tokens(raw_output)
            self.tokens_used_today += tokens_used

            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": compressed_output})

            if self._should_summarize_conversation():
                self._summarize_and_trim_history()

            # Store in long-term memory — no token_count parameter
            self.memory.store(
                content=f"Q: {prompt}\nA: {compressed_output[:500]}",
                tags=["conversation"]
            )

            self.working_memory.add(
                step="User query",
                result=compressed_output[:150],
                confidence=0.8 if "Confidence: High" in raw_output else 0.6
            )

            self.metrics.log(
                prompt=prompt,
                response=compressed_output,
                tokens_used=tokens_used,
                latency_ms=(time.time() - start_time) * 1000,
                model="routed"
            )

            return compressed_output

        except Exception as e:
            self.metrics.log(
                prompt=prompt,
                response=f"Error: {str(e)}",
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                model="error",
                success=False
            )
            return f"⚠️ Error: {str(e)[:150]}"

    def get_usage(self) -> Dict:
        self._reset_daily_if_needed()
        return {
            "tokens_used_today": self.tokens_used_today,
            "daily_limit": self.daily_token_limit,
            "remaining": max(0, self.daily_token_limit - self.tokens_used_today),
            "reset_in_hours": max(0, 24 - (datetime.now() - self.last_reset).seconds // 3600)
        }

    def get_metrics(self) -> Dict:
        return self.metrics.get_report()

    def reset_working_memory(self):
        self.working_memory.clear()

    def export_conversation(self) -> List[Dict]:
        return self.conversation_history.copy()
