"""
Token-optimized, free-tier AI agent.
Multi-provider: Groq + Sambanova + Cerebras + NVIDIA NIM + Modal (all free tier).
Multi-turn conversation history, streaming, SQLite memory.
Native function/tool calling with 6 tools:
  web_search, run_python, read_file, fetch_url, recall_memory, calculate
"""
import os
import re
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Generator

from groq import Groq, RateLimitError as GroqRateLimitError
import tiktoken

from .caveman import CAVEMAN_SYSTEM_PROMPT, compress_response
from .memory import TokenEfficientMemory
from .router import ModelRouter, GROQ, SAMBANOVA, CEREBRAS, NVIDIA, MODAL, MINIMAX, OPENROUTER, TOGETHER

# Max turns to keep in live conversation context (before summarizing to memory)
MAX_HISTORY_TURNS = 12  # 12 pairs = 24 messages

# ── Tool definitions for function calling API ─────────────────────────────────
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information, news, facts, people, prices, "
                "or anything that may have changed recently. Use this whenever the user "
                "asks about something that requires up-to-date knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code and return the output. Use this to run calculations, "
                "test logic, process data, or verify that code works correctly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Valid Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file in the project directory. "
                "Use this to inspect source code, configs, or any project file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Relative path to the file such as src/agent.py or app.py"
                    },
                    "line_number": {
                        "type": "integer",
                        "description": "Optional specific line number to focus on"
                    }
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch and extract the text content of a webpage. "
                "Use this when the user provides a URL or asks to summarize a webpage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch, must include the protocol prefix"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Search long-term memory for relevant past conversations or stored facts. "
                "Use this when the user references something from a previous session "
                "or asks what was discussed before."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
              "type": "string",
              "description": "What to search for in memory"
            }
          },
          "required": [
            "topic"
          ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Perform precise mathematical calculations using sympy. "
                "Use this for algebra, calculus, equations, symbolic math, "
                "unit conversions, or any calculation requiring exact precision."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A sympy-compatible math expression or equation. "
                            "Examples: 'sqrt(144)', 'solve(x**2 - 4, x)', "
                            "'integrate(x**2, x)', 'simplify((x**2-1)/(x-1))'"
                        )
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "store_memory",
            "description": (
                "Store an important fact, preference, or piece of information in long-term memory. "
                "Use this when the user explicitly asks you to remember something, "
                "or when you learn something important about the user or their project "
                "that should be recalled in future sessions. "
                "After storing, confirm to the user what was saved in a friendly, concise message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact or information to store in memory"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags to categorize the memory e.g. ['user', 'project', 'preference']"
                    }
                },
                "required": ["content"]
            }
        }
    },
]

# Keyword fallback triggers used ONLY in deep reasoning mode
# (when DeepSeek-R1 is active and tool calling is not available)
_SEARCH_TRIGGERS = [
    "search for", "look up", "find me", "what is the latest",
    "latest", "current", "news about", "who is", "what happened",
    "today", "recently", "right now"
]


class FreeAgent:
    """100% free, token-optimized reasoning agent.

    Supports Groq, Sambanova, Cerebras — falls back automatically.
    Maintains real multi-turn conversation history within session.
    Persists long-term memory to SQLite across sessions.

    Modes:
    - Normal mode:       Tool-capable model (Groq 70B preferred).
                         Native function calling for all 6 tools.
    - Deep reasoning:    DeepSeek-R1 on Sambanova.
                         Falls back to keyword triggers (no native tool calling).
                         Memory stays fully consistent across mode switches.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        daily_token_limit: int = 50000,
        memory_path: str = "agent_memory.db",
        caveman_mode: bool = True,
        deep_reasoning_mode: bool = False,
    ):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.caveman_mode = caveman_mode
        self.deep_reasoning_mode = deep_reasoning_mode
        self._last_model_label = "⚡ 8B"
        self._last_provider = "groq"

        # ── Provider clients ──────────────────────────────────────────────
        groq_key = api_key or os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError(
                "GROQ_API_KEY required. Get free key: https://console.groq.com/keys"
            )
        self.groq_client = Groq(api_key=groq_key, max_retries=0)

        # Sambanova
        sambanova_key = os.getenv("SAMBANOVA_API_KEY")
        self.sambanova_client = None
        if sambanova_key:
            try:
                from openai import OpenAI
                self.sambanova_client = OpenAI(
                    api_key=sambanova_key,
                    base_url="https://api.sambanova.ai/v1"
                )
            except ImportError:
                pass

        # Cerebras
        cerebras_key = os.getenv("CEREBRAS_API_KEY")
        self.cerebras_client = None
        if cerebras_key:
            try:
                from cerebras.cloud.sdk import Cerebras
                self.cerebras_client = Cerebras(api_key=cerebras_key)
            except ImportError:
                try:
                    from openai import OpenAI
                    self.cerebras_client = OpenAI(
                        api_key=cerebras_key,
                        base_url="https://api.cerebras.ai/v1"
                    )
                except ImportError:
                    pass

        # Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = None
        if gemini_key:
            try:
                from openai import OpenAI
                self.gemini_client = OpenAI(
                    api_key=gemini_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
            except ImportError:
                pass

        # OpenRouter
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_client = None
        if openrouter_key:
            try:
                from openai import OpenAI
                self.openrouter_client = OpenAI(
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1"
                )
            except ImportError:
                pass

        # NVIDIA NIM (Nemotron-3-Nano-30B — tool calling fallback1)
        nvidia_nemotron_key = os.getenv("NVIDIA_API_KEY_NEMOTRON")
        self.nvidia_client = None
        if nvidia_nemotron_key:
            try:
                from openai import OpenAI
                self.nvidia_client = OpenAI(
                    api_key=nvidia_nemotron_key,
                    base_url="https://integrate.api.nvidia.com/v1"
                )
            except ImportError:
                pass

        # Modal (GLM-5.1-FP8 — primary coding + deep reasoning)
        # Uses modalresearch_ key as simple Bearer token (from modal.com/glm-5-endpoint)
        modal_key = os.getenv("GLM51_MODAL_KEY")
        self.modal_client = None
        if modal_key:
            try:
                from openai import OpenAI
                self.modal_client = OpenAI(
                    api_key=modal_key,
                    base_url="https://api.us-west-2.modal.direct/v1"
                )
            except ImportError:
                pass
        # OpenRouter (GLM-5.1 — permanent fallback after Modal, credits-based)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_glm_client = None
        if openrouter_key:
            try:
                from openai import OpenAI
                self.openrouter_glm_client = OpenAI(
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1"
                )
            except ImportError:
                pass
        # Together AI (GLM-5.1 — second permanent fallback, $25 free credits)
        together_key = os.getenv("TOGETHER_API_KEY")
        self.together_client = None
        if together_key:
            try:
                from openai import OpenAI
                self.together_client = OpenAI(
                    api_key=together_key,
                    base_url="https://api.together.xyz/v1"
                )
            except ImportError:
                pass

        # MiniMax M2.7 (via NVIDIA NIM — general reasoning, free after eval testing)
        minimax_key = os.getenv("NVIDIA_API_KEY")
        self.minimax_client = None
        if minimax_key:
            try:
                from openai import OpenAI
                self.minimax_client = OpenAI(
                    api_key=minimax_key,
                    base_url="https://integrate.api.nvidia.com/v1"
                )
            except ImportError:
                pass

        # Tavily (web search — used in both tool calling and keyword fallback)
        self.tavily_key = os.getenv("TAVILY_API_KEY")

        # ── Budget tracking ───────────────────────────────────────────────
        self.daily_token_limit = daily_token_limit
        self.tokens_used_today = 0
        self.last_reset = datetime.now()

        # ── Components ────────────────────────────────────────────────────
        self.memory = TokenEfficientMemory(memory_path)
        self.router = ModelRouter()

        # Multi-turn conversation history — shared across ALL modes
        self.conversation_history: List[Dict] = []

        # ── Available models ──────────────────────────────────────────────
        self.available_models: Dict[str, dict] = {}
        self._register_available_models()

    # ── FILE READING ──────────────────────────────────────────────────────────
    def read_file(self, filepath: str, line_number: int = None) -> str:
        """Read a file from the project directory."""
        try:
            current_file = os.path.abspath(__file__)
            base_dir = os.path.dirname(os.path.dirname(current_file))
            full_path = os.path.join(base_dir, filepath)

            if not os.path.exists(full_path):
                return f"Error: File '{filepath}' not found."

            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if line_number is None:
                return "".join(lines)

            if line_number < 1 or line_number > len(lines):
                return f"Error: Line {line_number} out of range. File has {len(lines)} lines."

            start = max(0, line_number - 6)
            end = min(len(lines), line_number + 5)
            output = [f"📄 Showing lines {start+1}-{end} of {filepath}:\n\n```python\n"]
            for i in range(start, end):
                line_num = i + 1
                content = lines[i].rstrip()
                if line_num == line_number:
                    output.append(f"{line_num:4d} >>> {content}\n")
                else:
                    output.append(f"{line_num:4d}     {content}\n")
            output.append("```")
            return "".join(output)

        except Exception as e:
            return f"Error reading file: {str(e)}"

    # ── FILE EDITING ──────────────────────────────────────────────────────────
    def edit_file(self, filepath: str, new_content: str) -> str:
        """Safely edit a file with backup and syntax validation."""
        try:
            if ".." in filepath or filepath.startswith("/"):
                return "⛔ Security Error: Invalid file path."

            allowed_extensions = ['.py', '.txt', '.md', '.json', '.yaml', '.env']
            if not any(filepath.endswith(ext) for ext in allowed_extensions):
                return f"⛔ Security Error: Cannot edit {filepath}."

            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(base_dir, filepath)

            if not os.path.exists(full_path):
                return f"Error: File '{filepath}' not found."

            backup_path = full_path + ".bak"
            with open(full_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)

            if filepath.endswith('.py'):
                try:
                    compile(new_content, full_path, 'exec')
                except SyntaxError as e:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    return f"⛔ Syntax Error: Line {e.lineno}: {e.msg}\n💾 Original preserved."

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return f"✅ Updated `{filepath}`.\n💾 Backup at `{filepath}.bak`\n🛡️ Syntax validated."

        except Exception as e:
            return f"❌ Error editing file: {str(e)}"

    # ── MODEL REGISTRATION ────────────────────────────────────────────────────
    def _register_available_models(self):
        """Register provider+model combos we have keys for."""
        self.available_models["llama-3.1-8b-instant"] = {"provider": GROQ, "rpd": 14400}
        self.available_models["llama-3.3-70b-versatile"] = {"provider": GROQ, "rpd": 1000}

        if self.sambanova_client:
            self.available_models["Meta-Llama-3.1-8B-Instruct"] = {"provider": SAMBANOVA, "rpd": 10000}
            self.available_models["Meta-Llama-3.3-70B-Instruct"] = {"provider": SAMBANOVA, "rpd": 5000}
            self.available_models["DeepSeek-R1"] = {"provider": SAMBANOVA, "rpd": 5000}

        if self.cerebras_client:
            self.available_models["llama3.1-8b"] = {"provider": CEREBRAS, "rpd": 50000}
            self.available_models["qwen-3-235b-a22b-instruct-2507"] = {"provider": CEREBRAS, "rpd": 50000}

        if self.nvidia_client:
            self.available_models["nvidia/nemotron-3-nano-30b-a3b"] = {"provider": NVIDIA, "rpd": 10000}

        if self.modal_client:
            self.available_models["zai-org/GLM-5.1-FP8"] = {"provider": MODAL, "rpd": 10000}
        if self.openrouter_glm_client:
            self.available_models["z-ai/glm-5.1"] = {"provider": OPENROUTER, "rpd": 10000}
        if self.together_client:
            self.available_models["z-ai/glm-5.1-together"] = {"provider": TOGETHER, "rpd": 10000}
        if self.minimax_client:
            self.available_models["minimaxai/minimax-m2.7"] = {"provider": MINIMAX, "rpd": 10000}

    def _reset_daily_if_needed(self):
        now = datetime.now()
        if now - self.last_reset > timedelta(hours=24):
            self.tokens_used_today = 0
            self.last_reset = now

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    # ── TOOL EXECUTORS ────────────────────────────────────────────────────────
    def _tool_web_search(self, query: str) -> str:
        """Execute web_search tool."""
        return self.tavily_search(query) or f"No results found for: {query}"

    def _tool_run_python(self, code: str) -> str:
        """Execute run_python tool via SafeCodeRunner."""
        try:
            from src.code_runner import SafeCodeRunner
            result = SafeCodeRunner.run(code)
            if result['success']:
                output = result.get('output', '').strip()
                ms = result.get('execution_time_ms', 0)
                return f"✅ Executed in {ms}ms:\n{output}" if output else f"✅ Executed in {ms}ms (no output)"
            else:
                return f"❌ Error: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"❌ Code runner failed: {str(e)}"

    def _tool_read_file(self, filepath: str, line_number: int = None) -> str:
        """Execute read_file tool."""
        known_src_files = ['agent.py', 'caveman.py', 'memory.py', 'router.py', 'code_runner.py']
        if "/" not in filepath and filepath in known_src_files:
            filepath = "src/" + filepath
        return self.read_file(filepath, line_number=line_number)

    def _tool_fetch_url(self, url: str) -> str:
        """Execute fetch_url tool."""
        return self._fetch_url_content(url)

    def _tool_recall_memory(self, query: str) -> str:
        """Execute recall_memory tool."""
        recalled = self.memory.recall(query, max_tokens=1200)
        return recalled if recalled else "No relevant memories found."

    def _tool_calculate(self, expression: str) -> str:
        """Execute calculate tool using sympy for precise math."""
        try:
            import sympy
            from sympy.parsing.sympy_parser import (
                parse_expr,
                standard_transformations,
                implicit_multiplication_application,
            )

            transformations = standard_transformations + (implicit_multiplication_application,)
            local_dict = {name: getattr(sympy, name) for name in dir(sympy) if not name.startswith('_')}

            # Handle common function calls like solve(), integrate(), diff()
            # by evaluating with sympy's full namespace
            try:
                result = eval(expression, {"__builtins__": {}}, local_dict)
                # Simplify if possible
                if hasattr(result, 'simplify'):
                    result = sympy.simplify(result)
                return f"Result: {result}"
            except Exception:
                # Fallback: try parse_expr for pure expressions
                expr = parse_expr(expression, transformations=transformations, local_dict=local_dict)
                result = sympy.simplify(expr)
                return f"Result: {result}"

        except ImportError:
            # sympy not installed — fall back to safe Python eval
            try:
                import math
                safe_globals = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
                safe_globals['__builtins__'] = {}
                result = eval(expression, safe_globals)
                return f"Result: {result} (sympy not installed, used math module)"
            except Exception as e:
                return f"❌ Calculation error: {str(e)}"
        except Exception as e:
            return f"❌ Calculation error: {str(e)}"

    def _execute_tool_call(self, tool_name: str, tool_args: dict) -> str:
        """
        Central dispatcher: routes tool_name → correct tool method.
        Returns string result to feed back to the model.
        """
        print(f"[ToolCall] {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:120]})", flush=True)

        if tool_name == "web_search":
            return self._tool_web_search(tool_args.get("query", ""))

        elif tool_name == "run_python":
            return self._tool_run_python(tool_args.get("code", ""))

        elif tool_name == "read_file":
            return self._tool_read_file(
                tool_args.get("filepath", ""),
                tool_args.get("line_number", None)
            )

        elif tool_name == "fetch_url":
            return self._tool_fetch_url(tool_args.get("url", ""))

        elif tool_name == "recall_memory":
            return self._tool_recall_memory(tool_args.get("topic", ""))

        elif tool_name == "calculate":
            return self._tool_calculate(tool_args.get("expression", ""))

        elif tool_name == "store_memory":
            content = tool_args.get("content", "")
            tags = tool_args.get("tags", ["explicit"])
            if not content:
                return "❌ No content provided to store."
            mem_id = self.memory.store(content=content, tags=tags)
            return (
                f"✅ Memory stored successfully. "
                f"Content: '{content[:80]}{'...' if len(content) > 80 else ''}'. "
                f"You should now confirm to the user that you have remembered this information."
            )
        else:
            return f"❌ Unknown tool: {tool_name}"

  # ── REVIEW PASS ───────────────────────────────────────────────────────────
    def _review_pass(self, answer: str, model: str, provider: str, max_tokens: int) -> str:
        """
        Lightweight self-review: check if the answer is complete and accurate.
        Returns improved answer if issues found, original answer otherwise.
        Adds ~1 API call latency only on tool-using queries.
        """
        try:
            review_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a reviewer. Your job is to check if an AI answer is complete, "
                        "accurate, and directly addresses what was asked. "
                        "If the answer is good, reply with exactly: LGTM\n"
                        "If the answer has issues (missing info, wrong facts, incomplete), "
                        "reply with the corrected and complete answer only — no preamble."
                    )
                },
                {
                    "role": "user",
                    "content": f"Review this answer and either reply LGTM or provide a corrected version:\n\n{answer}"
                }
            ]
            review_response = self._call_provider(
                model=model,
                provider=provider,
                messages=review_messages,
                max_tokens=max_tokens // 2,
                stream=False,
            )
            reviewed = review_response.choices[0].message.content or ""
            if reviewed.strip().upper().startswith("LGTM"):
                return answer  # original is good
            elif len(reviewed.strip()) > 20:
                print(f"[ReviewPass] Answer improved by reviewer", flush=True)
                return reviewed  # use improved version
            else:
                return answer  # fallback to original if review is too short
        except Exception as e:
            print(f"[ReviewPass] Failed: {e} — using original answer", flush=True)
            return answer  # always safe — never breaks the response
  
    # ── NATIVE TOOL CALLING LOOP (Normal mode) ────────────────────────────────
    def _run_tool_calling_loop(
        self,
        messages: List[Dict],
        model: str,
        provider: str,
        max_output_tokens: int,
        stream: bool = False,
    ):
        """
        Runs the full agentic tool-calling loop (non-streaming).

        1. Call model with tools
        2. If model calls a tool → execute it → append result → repeat
        3. When model gives final text answer → return it

        Max 5 tool rounds to prevent infinite loops.
        Returns (final_text, used_model, used_provider)
        """
        MAX_TOOL_ROUNDS = 5
        current_messages = list(messages)
        seen_tool_calls = set()  # loop detection: track (tool_name, args) pairs
        # Fallback chain for rate limits — primary → 8B → Cerebras → SambaNova
        tool_models_to_try = [(model, provider)]
        if model != "llama-3.1-8b-instant":
            tool_models_to_try.append(("llama-3.1-8b-instant", GROQ))
        tool_models_to_try.append(("qwen-3-235b-a22b-instruct-2507", CEREBRAS))
        tool_models_to_try.append(("Meta-Llama-3.3-70B-Instruct", SAMBANOVA))

        for round_num in range(MAX_TOOL_ROUNDS):
            # Non-streaming call to check for tool use — with rate-limit fallback
            response = None
            for try_model, try_provider in tool_models_to_try:
                try:
                    response = self._call_provider(
                        model=try_model,
                        provider=try_provider,
                        messages=current_messages,
                        max_tokens=max_output_tokens,
                        stream=False,
                        tools=TOOL_DEFINITIONS,
                        tool_choice="auto",
                    )
                    model, provider = try_model, try_provider
                    break
                except Exception as e:
                    err = str(e).lower()
                    if isinstance(e, GroqRateLimitError) or "429" in err or "rate_limit" in err or "tool_use_failed" in err or "failed_generation" in err or "413" in err or "request too large" in err or "tokens per minute" in err:
                        continue
                    raise
            if response is None:
                yield "⚠️ All providers rate-limited. Try again later."
                return

            choice = response.choices[0]
            message = choice.message

            # No tool calls — model gave final answer
            if not message.tool_calls:
                return message.content or "", model, provider

            # Model wants to call tools — execute each one
            # Append assistant message with tool_calls to history
            current_messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Execute each tool call and append results
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                # Loop detection — skip duplicate tool+args combinations
                call_signature = (tc.function.name, tc.function.arguments.strip())
                if call_signature in seen_tool_calls:
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            "[Loop detected] You already called this tool with these exact arguments. "
                            "Do not repeat it. Synthesize your answer from what you already have."
                        ),
                    })
                    continue
                seen_tool_calls.add(call_signature)

                tool_result = self._execute_tool_call(tc.function.name, args)

                # Self-correction: enrich error results with guidance
                is_error = any(marker in tool_result for marker in [
                    "Error", "error", "❌", "failed", "not found", "No results"
                ])
                if is_error:
                    tool_result = (
                        f"{tool_result}\n\n"
                        "[Self-correction] This tool call did not succeed. "
                        "Do NOT repeat the same call. "
                        "Try a different query, a different tool, or reason from what you already know."
                    )

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })
        # Exhausted rounds — ask model to summarize what it has
        current_messages.append({
            "role": "user",
            "content": "Please summarize what you found and give your final answer."
        })
        final = self._call_provider(
            model=model,
            provider=provider,
            messages=current_messages,
            max_tokens=max_output_tokens,
            stream=False,
        )
        return final.choices[0].message.content or "", model, provider

    def _run_tool_calling_loop_stream(
        self,
        messages: List[Dict],
        model: str,
        provider: str,
        max_output_tokens: int,
    ) -> Generator[str, None, None]:
        """
        Streaming tool loop with rate-limit fallback.
        Tool rounds are non-streaming (fast). Only final answer is streamed.
        """
        MAX_TOOL_ROUNDS = 5
        current_messages = list(messages)
        seen_tool_calls = set()  # loop detection: track (tool_name, args) pairs
        tool_models_to_try = [(model, provider)]
        if model != "llama-3.1-8b-instant":
            tool_models_to_try.append(("llama-3.1-8b-instant", GROQ))
        tool_models_to_try.append(("qwen-3-235b-a22b-instruct-2507", CEREBRAS))
        tool_models_to_try.append(("Meta-Llama-3.3-70B-Instruct", SAMBANOVA))

        for round_num in range(MAX_TOOL_ROUNDS):
            response = None
            for try_model, try_provider in tool_models_to_try:
                try:
                    response = self._call_provider(
                        model=try_model,
                        provider=try_provider,
                        messages=current_messages,
                        max_tokens=max_output_tokens,
                        stream=False,
                        tools=TOOL_DEFINITIONS,
                        tool_choice="auto",
                    )
                    model, provider = try_model, try_provider
                    break
                except Exception as e:
                    err = str(e).lower()
                    if isinstance(e, GroqRateLimitError) or "429" in err or "rate_limit" in err or "tool_use_failed" in err or "failed_generation" in err or "413" in err or "request too large" in err or "tokens per minute" in err:
                        continue
                    raise
            if response is None:
                yield "⚠️ All providers rate-limited. Try again later."
                return

            choice = response.choices[0]
            message = choice.message
            # No tool calls — stream the final answer
            if not message.tool_calls:
                if round_num == 0:
                    # No tools used at all — stream directly
                    final_text = message.content or ""
                    words = final_text.split(" ")
                    for i, word in enumerate(words):
                        yield word + (" " if i < len(words) - 1 else "")
                    return
                else:
                    # Tools were used — force synthesis from observations
                    current_messages.append({
                        "role": "user",
                        "content": (
                            "Using ONLY the tool results in your Observations above, "
                            "give your final answer now. "
                            "Do not recalculate or second-guess the tool results — use them exactly as returned."
                        )
                    })
                    for syn_model, syn_provider in tool_models_to_try:
                        try:
                            final_stream = self._call_provider(
                                model=syn_model,
                                provider=syn_provider,
                                messages=current_messages,
                                max_tokens=max_output_tokens,
                                stream=False,
                            )
                            final_text = final_stream.choices[0].message.content or ""
                            self._last_provider = syn_provider
                            self._last_model_label = {
                                "qwen-3-235b-a22b-instruct-2507": "⚡ Qwen3",
                                "Meta-Llama-3.3-70B-Instruct": "🔥 70B",
                                "llama-3.1-8b-instant": "⚡ 8B",
                                "llama-3.3-70b-versatile": "🔥 70B",
                            }.get(syn_model, "⚡ 8B")
                            # Review pass — lightweight self-check
                            final_text = self._review_pass(final_text, syn_model, syn_provider, max_output_tokens)
                            yield final_text
                            break
                        except Exception as e:
                            err = str(e).lower()
                            if isinstance(e, GroqRateLimitError) or "429" in err or "rate_limit" in err or "413" in err or "request too large" in err or "tokens per minute" in err:
                                continue
                            raise
                    return

            # Enforce one tool per round — take only the first tool call
            tool_calls_this_round = message.tool_calls[:1]
            tool_names = [tc.function.name for tc in tool_calls_this_round]
            yield f"\n\n🔧 *Using tools: {', '.join(tool_names)}...*\n\n"
            # Append assistant message with tool_calls
            current_messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in tool_calls_this_round
                ]
            })
            # Execute each tool and append result
            for tc in tool_calls_this_round:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                # Loop detection — skip duplicate tool+args combinations
                call_signature = (tc.function.name, tc.function.arguments.strip())
                if call_signature in seen_tool_calls:
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            "[Loop detected] You already called this tool with these exact arguments. "
                            "Do not repeat it. Synthesize your answer from what you already have."
                        ),
                    })
                    continue
                seen_tool_calls.add(call_signature)

                tool_result = self._execute_tool_call(tc.function.name, args)

                # Self-correction: enrich error results with guidance
                is_error = any(marker in tool_result for marker in [
                    "Error", "error", "❌", "failed", "not found", "No results"
                ])
                if is_error:
                    tool_result = (
                        f"{tool_result}\n\n"
                        "[Self-correction] This tool call did not succeed. "
                        "Do NOT repeat the same call. "
                        "Try a different query, a different tool, or reason from what you already know."
                    )

                if tc.function.name == "run_python":
                    yield f"`{tool_result}`\n\n"

                observation = f"Observation (round {round_num + 1}, tool={tc.function.name}): {tool_result}"
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation,
                })

        # Exhausted MAX_TOOL_ROUNDS — stream final synthesis
        current_messages.append({
            "role": "user",
            "content": (
                "You have completed all reasoning rounds. "
                "Based on all your Observations above, give your final answer now. "
                "Do not call any more tools."
            )
        })
        for syn_model, syn_provider in tool_models_to_try:
            try:
                final_response = self._call_provider(
                    model=syn_model,
                    provider=syn_provider,
                    messages=current_messages,
                    max_tokens=max_output_tokens,
                    stream=False,
                )
                final_text = final_response.choices[0].message.content or ""
                self._last_provider = syn_provider
                self._last_model_label = {
                    "qwen-3-235b-a22b-instruct-2507": "⚡ Qwen3",
                    "Meta-Llama-3.3-70B-Instruct": "🔥 70B",
                    "llama-3.1-8b-instant": "⚡ 8B",
                    "llama-3.3-70b-versatile": "🔥 70B",
                }.get(syn_model, "⚡ 8B")
                # Review pass — lightweight self-check
                final_text = self._review_pass(final_text, syn_model, syn_provider, max_output_tokens)
                yield final_text
                break
            except Exception as e:
                err = str(e).lower()
                if isinstance(e, GroqRateLimitError) or "429" in err or "rate_limit" in err or "413" in err or "request too large" in err or "tokens per minute" in err:
                    continue
                raise

    # ── URL FETCHING ──────────────────────────────────────────────────────────
    def _fetch_url_content(self, url: str) -> str:
        """Fetch and clean text content from a URL."""
        try:
            import requests
            from bs4 import BeautifulSoup

            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator=' ', strip=True)
            return text[:2000] if len(text) > 2000 else text

        except Exception as e:
            return f"Error fetching URL: {str(e)}"

    # ── MEMORY CONTEXT ────────────────────────────────────────────────────────
    def _get_memory_context(self, prompt: str) -> str:
        """Retrieve relevant long-term memories + any URL in prompt."""
        recalled = self.memory.recall(prompt, max_tokens=800)

        url_context = ""
        if "http://" in prompt or "https://" in prompt:
            urls = re.findall(r'https?://[^\s<>"]+', prompt)
            if urls:
                fetched = self._fetch_url_content(urls[0])
                if not fetched.startswith("Error"):
                    url_context = f"\n\nContent from {urls[0]}:\n{fetched}"

        return recalled + url_context

    # ── WEB SEARCH (Tavily) ───────────────────────────────────────────────────
    def tavily_search(self, query: str) -> str:
        """Search the web via Tavily."""
        if not self.tavily_key:
            return ""
        try:
            import requests
            r = requests.post("https://api.tavily.com/search", json={
                "api_key": self.tavily_key,
                "query": query,
                "max_results": 5
            }, timeout=5)
            results = r.json().get("results", [])
            if not results:
                return ""
            parts = ["🌐 Web Search Results:"]
            for i, res in enumerate(results, 1):
                parts.append(
                    f"{i}. {res.get('title','')}\n"
                    f"{res.get('url','')}\n"
                    f"{res.get('content','')[:900]}"
                )
            return "\n\n".join(parts)
        except Exception as e:
            print(f"[Tavily] Search failed: {e}", flush=True)
            return ""

    # ── FUNCTION EXTRACTION ───────────────────────────────────────────────────
    def _extract_function(self, source: str, func_name: str) -> str:
        """Extract a specific function from Python source by name."""
        lines = source.splitlines()
        result = []
        in_func = False
        func_indent = 0

        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if not in_func:
                if (stripped.startswith(f"def {func_name}(")
                        or stripped.startswith(f"async def {func_name}(")):
                    in_func = True
                    func_indent = indent
                    result.append(line)
            else:
                if (stripped and indent <= func_indent
                        and (stripped.startswith("def ")
                             or stripped.startswith("class ")
                             or stripped.startswith("async def "))):
                    break
                result.append(line)

        if result:
            return "\n".join(result)
        return source[:2000] + "\n... [function not found — showing file start]"

    # ── MULTI-FILE CONTEXT ────────────────────────────────────────────────────
    def _get_multi_file_context(self, prompt: str) -> str:
        """Smartly inject relevant code snippets into LLM context."""
        prompt_lower = prompt.lower()

        trigger_words = [
            "bug", "error", "fix", "broken", "not working", "issue", "debug",
            "feature", "add", "update",
            "not saving", "isn't", "doesn't",
            "caveman", "token", "compress", "mode", "routing", "router",
            "memory", "stream", "context", "inject",
            "code_runner", "code runner",
            "app.py", "agent.py", "caveman.py", "memory.py", "router.py",
        ]

        triggered = any(word in prompt_lower for word in trigger_words) and "[no-context]" not in prompt_lower
        print(f"[MultiFileContext] Trigger detected: {triggered}", flush=True)

        if not triggered:
            return ""

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        keyword_focus = [
            ("caveman",         "src/caveman.py",       None),
            ("compress",        "src/caveman.py",       None),
            ("token",           "src/caveman.py",       None),
            ("_build_messages", "src/agent.py",         "_build_messages"),
            ("deep reasoning",  "src/agent.py",         "_build_messages"),
            ("multi_file",      "src/agent.py",         "_get_multi_file_context"),
            ("context inject",  "src/agent.py",         "_get_multi_file_context"),
            ("routing",         "src/router.py",        None),
            ("router",          "src/router.py",        None),
            ("memory",          "src/memory.py",        None),
            ("stream",          "src/agent.py",         "ask_stream"),
            ("code_runner",     "src/code_runner.py",   None),
            ("code runner",     "src/code_runner.py",   None),
        ]

        files_to_read: dict = {}
        for keyword, filepath, func_name in keyword_focus:
            if keyword in prompt_lower:
                if filepath not in files_to_read or files_to_read[filepath] is None:
                    files_to_read[filepath] = func_name

        if not files_to_read:
            files_to_read = {
                "app.py":          None,
                "src/agent.py":    None,
                "src/caveman.py":  None,
            }

        print(f"[MultiFileContext] Files to read: {list(files_to_read.keys())}", flush=True)

        context_parts = [
            "--- ACTUAL PROJECT SOURCE CODE ---\n"
            "Reference specific function names, variable names, and exact values.\n"
            "Never give generic advice that ignores what is actually written here.\n"
            "---"
        ]
        total_chars = 0

        for filepath, func_name in files_to_read.items():
            full_path = os.path.join(base_dir, filepath)
            if not os.path.exists(full_path):
                print(f"[MultiFileContext] WARNING: {full_path} not found!", flush=True)
                continue

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    raw = f.read()

                if func_name:
                    content = self._extract_function(raw, func_name)
                    label = f"{filepath} → {func_name}()"
                else:
                    MAX_CHARS = 9000
                    if filepath == "src/agent.py" and len(raw) > MAX_CHARS:
                        key_funcs = ["ask_stream", "_build_messages", "_get_multi_file_context"]
                        parts = [self._extract_function(raw, f) for f in key_funcs]
                        content = "\n\n".join(p for p in parts if p)
                        label = f"{filepath} → [ask_stream, _build_messages, _get_multi_file_context]"
                    elif len(raw) <= MAX_CHARS:
                        content = raw
                    else:
                        content = raw[:MAX_CHARS] + f"\n... [truncated at {MAX_CHARS} chars]"
                    label = filepath

                context_parts.append(f"\n### {label}\n```python\n{content}\n```")
                total_chars += len(content)

            except Exception as e:
                print(f"[MultiFileContext] Error reading {filepath}: {e}", flush=True)

        print(f"[MultiFileContext] Total context chars injected: {total_chars}", flush=True)
        context_parts.append("--- END PROJECT CONTEXT ---")
        return "\n".join(context_parts) if len(context_parts) > 2 else ""

    # ── MESSAGE BUILDER ───────────────────────────────────────────────────────
    def _build_messages(self, prompt: str, memory_context: str) -> List[Dict]:
        """Build the full messages array for the API call."""
        if self.caveman_mode:
            from .caveman import CAVEMAN_SYSTEM_PROMPT
            system_content = CAVEMAN_SYSTEM_PROMPT
        elif self.deep_reasoning_mode:
            system_content = (
                "You are an expert AI assistant. You ALWAYS think step-by-step before answering.\n"
                "For every query, analyze constraints, edge cases, and logic internally.\n"
                "Provide clear, structured, and thorough answers. Use Markdown for formatting.\n"
                "If the user asks for code, explain the logic first, then provide the solution.\n"
                "Prioritize accuracy and depth over brevity.\n"
                "When web search results are provided in context, summarize naturally. "
                "Never list URLs unless explicitly asked.\n"
                "Before answering, write your reasoning inside <think></think> tags. "
                "Break the problem into steps. Check your logic. Then give your final answer."
            )
        else:
            system_content = (
                "You are a helpful, harmless, and honest AI assistant. "
                "Answer clearly and concisely, but provide detail when needed. "
                "When web search results or tool results are provided in context, "
                "summarize naturally without listing URLs unless asked. \n\n"

                "PLANNING RULE:\n"
                "Before calling ANY tool, you MUST write a line starting with 'Plan:' "
                "followed by what you need to find and which tools you will use in what order. "
                "Example: 'Plan: I will call web_search to find X, then calculate to compute Y.' "
                "NEVER make a tool call without writing your Plan: line first.\n\n"

                "TOOL DISCIPLINE:\n"
                "Call only one tool per round — never batch multiple tools together. "
                "Use each tool at most once per query. "
                "Do not repeat the same tool call with the same or similar arguments. "
                "If a tool returns an error or empty result, switch strategy — "
                "try a different query, a different tool, or reason from what you already know.\n\n"

                "LOOP GUARD:\n"
                "If you have called 3 or more tools and still lack a clear answer, "
                "stop calling tools and synthesize the best answer from what you have. "
                "Never call the same tool with the same arguments twice.\n\n"

                "SYNTHESIS:\n"
                "After receiving tool results, reason about what you learned. "
                "When you have enough information, synthesize all observations "
                "into a clear, direct final answer.\n\n"

                "TOOL ENFORCEMENT:\n"
                "If the user explicitly says 'Run this code', 'run it', 'debug it', 'Calculate', 'Search for', "
                "'Look up', or 'Find me', you MUST call the appropriate tool even if you "
                "already know the answer. Never skip a tool call the user has explicitly requested.\n\n"
            )

        messages = [{"role": "system", "content": system_content}]

        multi_file_context = self._get_multi_file_context(prompt)
        if multi_file_context:
            messages.append({"role": "system", "content": multi_file_context})

        if memory_context:
            messages.append({"role": "system", "content": f"Relevant past context:\n{memory_context}"})

        history_to_include = self.conversation_history[-MAX_HISTORY_TURNS * 2:]
        messages.extend(history_to_include)

        messages.append({"role": "user", "content": prompt})
        return messages

    # ── PROVIDER DISPATCHER ───────────────────────────────────────────────────
    def _call_provider(
        self,
        model: str,
        provider: str,
        messages: List[Dict],
        max_tokens: int,
        stream: bool = False,
        tools: list = None,
        tool_choice: str = None,
    ):
        """Dispatch API call to the correct provider client."""
        kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.95,
            stream=stream,
        )

        # Only attach tools if the model supports function calling
        if tools and self.router.supports_tool_calling(model):
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        if provider == GROQ:
            return self.groq_client.chat.completions.create(**kwargs)
        elif provider == SAMBANOVA:
            if not self.sambanova_client:
                raise RuntimeError("Sambanova client not initialized")
            return self.sambanova_client.chat.completions.create(**kwargs)
        elif provider == CEREBRAS:
            if not self.cerebras_client:
                raise RuntimeError("Cerebras client not initialized")
            return self.cerebras_client.chat.completions.create(**kwargs)
        elif provider == NVIDIA:
            if not self.nvidia_client:
                raise RuntimeError("NVIDIA client not initialized")
            return self.nvidia_client.chat.completions.create(**kwargs)
        elif provider == MODAL:
            if not self.modal_client:
                raise RuntimeError("Modal client not initialized")
            return self.modal_client.chat.completions.create(**kwargs)
        elif provider == MINIMAX:
            if not self.minimax_client:
                raise RuntimeError("MiniMax client not initialized")
            return self.minimax_client.chat.completions.create(**kwargs)
        elif provider == OPENROUTER:
            if not self.openrouter_glm_client:
                raise RuntimeError("OpenRouter client not initialized")
            return self.openrouter_glm_client.chat.completions.create(**kwargs)
        elif provider == TOGETHER:
            if not self.together_client:
                raise RuntimeError("Together client not initialized")
            return self.together_client.chat.completions.create(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # ── HISTORY MANAGEMENT ────────────────────────────────────────────────────
    def _append_to_history(self, prompt: str, response: str):
        """Append turn to conversation history, summarizing overflow to memory."""
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": response})

        if len(self.conversation_history) > MAX_HISTORY_TURNS * 2:
            dropped = self.conversation_history[:2]
            summary = f"Q: {dropped[0]['content']}\nA: {dropped[1]['content']}"
            self.memory.store(content=summary, tags=["conversation"], token_count=None)
            self.conversation_history = self.conversation_history[2:]

    # ── PUBLIC API: ask() ─────────────────────────────────────────────────────
    def ask(self, prompt: str, max_output_tokens: int = 4096) -> str:
        """
        Main entry point (non-streaming).

        Normal mode:        Tool-capable model + native function calling.
        Deep reasoning mode: DeepSeek-R1 + keyword trigger fallback.
        Memory is always consistent across mode switches.
        """
        self._reset_daily_if_needed()
        if self.tokens_used_today >= self.daily_token_limit:
            return "⚠️ Daily token limit reached. Resets in 24h."

        memory_context = self._get_memory_context(prompt)
        messages = self._build_messages(prompt, memory_context)

        raw_output = None
        used_model = None
        used_provider = None

        # ── NORMAL MODE: native tool calling ─────────────────────────────
        if not self.deep_reasoning_mode:
            model, provider = self.router.select_tool_capable_model(prompt, self.available_models)
            model_label = self.router.get_complexity_label(prompt)
            self._last_model_label = model_label
            self._last_provider = provider

            providers_to_try = [(model, provider)]
            if provider != GROQ:
                providers_to_try.append(("llama-3.3-70b-versatile", GROQ))
            providers_to_try.append(("qwen-3-235b-a22b-instruct-2507", CEREBRAS))
            providers_to_try.append(("nvidia/nemotron-3-nano-30b-a3b", NVIDIA))
            providers_to_try.append(("Meta-Llama-3.3-70B-Instruct", SAMBANOVA))
            providers_to_try.append(("llama-3.1-8b-instant", GROQ))

            for try_model, try_provider in providers_to_try:
                try:
                    raw_output, used_model, used_provider = self._run_tool_calling_loop(
                        messages=messages,
                        model=try_model,
                        provider=try_provider,
                        max_output_tokens=max_output_tokens,
                    )
                    break
                except Exception as e:
                    last_error = str(e)
                    continue

        # ── DEEP REASONING MODE: DeepSeek-R1 + keyword fallback ──────────
        else:
            model, provider = self.router.select_model(
                prompt, self.available_models, force_reasoning=True
            )
            model_label = self.router.get_complexity_label(prompt)
            self._last_model_label = model_label
            self._last_provider = provider

            # Keyword trigger fallback (DeepSeek-R1 has no tool calling)
            if any(t in prompt.lower() for t in _SEARCH_TRIGGERS):
                search_result = self.tavily_search(prompt)
                if search_result:
                    memory_context = f"{search_result}\n\n{memory_context}".strip()
                    messages = self._build_messages(prompt, memory_context)

            providers_to_try = [(model, provider)]
            if provider != GROQ:
                providers_to_try.append(("llama-3.3-70b-versatile", GROQ))
            providers_to_try.append(("qwen-3-235b-a22b-instruct-2507", CEREBRAS))
            providers_to_try.append(("nvidia/nemotron-3-nano-30b-a3b", NVIDIA))
            providers_to_try.append(("Meta-Llama-3.3-70B-Instruct", SAMBANOVA))
            providers_to_try.append(("llama-3.1-8b-instant", GROQ))

            for try_model, try_provider in providers_to_try:
                try:
                    response = self._call_provider(
                        model=try_model,
                        provider=try_provider,
                        messages=messages,
                        max_tokens=max_output_tokens,
                        stream=False,
                    )
                    raw_output = response.choices[0].message.content
                    used_model = try_model
                    used_provider = try_provider
                    tokens_used = getattr(response.usage, "total_tokens", 0)
                    self.tokens_used_today += tokens_used
                    break
                except Exception as e:
                    last_error = str(e)
                    continue

        if raw_output is None:
            return f"⚠️ All providers failed. Last error: {getattr(self, '_last_error', 'unknown')[:150]}"

        think_stripped = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
        if think_stripped:
            raw_output = think_stripped
        else:
            think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
            raw_output = think_match.group(1).strip() if think_match else raw_output
        output = compress_response(raw_output) if self.caveman_mode else raw_output

        self._append_to_history(prompt, output)
        self.tokens_used_today += self._count_tokens(output)

        output += f"\n\n`{model_label} · {used_provider}`"
        return output

    # ── PUBLIC API: ask_stream() ──────────────────────────────────────────────
    def ask_stream(self, prompt: str, max_output_tokens: int = 4096) -> Generator[str, None, None]:
        """
        Streaming entry point.

        Normal mode:        Tool-capable model + native function calling (stream final answer).
        Deep reasoning mode: DeepSeek-R1 + keyword trigger fallback (full streaming).
        Memory is always consistent across mode switches.
        """
        self._reset_daily_if_needed()
        if self.tokens_used_today >= self.daily_token_limit:
            yield "⚠️ Daily token limit reached. Resets in 24h."
            return

        # ── FILE READING INTERCEPTOR ──────────────────────────────────────
        simple_read_only = any(word in prompt.lower() for word in ["show me ", "check line", "what is on line", "line "])
        has_analysis_request = any(word in prompt.lower() for word in [
            "fix", "analyze", "analyse", "why", "how", "bug", "error",
            "tell me why", "explain", "understand", "caveman", "token",
            "compress", "mode", "routing", "router", "memory", "stream",
            "reasoning", "context", "not saving", "isn't", "doesn't",
        ])

        if simple_read_only and not has_analysis_request:
            match = re.search(r'([\w\./]+\.(py|txt|md|json|yaml|yml|env|toml))', prompt)
            if match:
                filepath = match.group(1)
                known_src_files = ['agent.py', 'caveman.py', 'memory.py', 'router.py']
                if "/" not in filepath and filepath in known_src_files:
                    filepath = "src/" + filepath
                line_match = re.search(r'line\s*(\d+)', prompt.lower())
                target_line = int(line_match.group(1)) if line_match else None
                file_content = self.read_file(filepath, line_number=target_line)
                if not file_content.startswith("Error"):
                    yield f"✅ **{'Line ' + str(target_line) + ' of' if target_line else 'Found file:'} `{filepath}`:**\n\n"
                    yield f"{file_content}\n"
                    return
                else:
                    yield f"⚠️ {file_content}\n\n"

        # ── FILE OPERATIONS INTERCEPTOR ───────────────────────────────────
        lower_prompt = prompt.lower()
        if any(word in lower_prompt for word in ["read file", "read ", "show me ", "check line", "what is on line", "edit ", "update ", "change "]):
            match = re.search(r'([\w\./]+\.(py|txt|md|json|yaml|yml|env|toml))', lower_prompt)
            if match:
                filepath = match.group(1)
                known_src_files = ['agent.py', 'caveman.py', 'memory.py', 'router.py']
                if "/" not in filepath and filepath in known_src_files:
                    filepath = "src/" + filepath

                line_match = re.search(r'line\s*(\d+)', lower_prompt)
                if line_match and any(w in lower_prompt for w in ["read", "show", "check"]):
                    result = self.read_file(filepath, line_number=int(line_match.group(1)))
                    yield f"{result}\n"
                    return

                if any(word in lower_prompt for word in ["edit", "update", "change", "replace"]):
                    code_match = re.search(r'```(?:python)?\s*(.*?)```', prompt, re.DOTALL)
                    if code_match:
                        result = self.edit_file(filepath, code_match.group(1).strip())
                        yield f"{result}\n"
                        return
                    else:
                        yield f"🛠️ I can edit `{filepath}` safely (with backup).\nProvide the **full new content** in a code block:\n\n```python\n# your new code here\n```\n"
                        return

                if not has_analysis_request:
                    file_content = self.read_file(filepath)
                    if not file_content.startswith("Error"):
                        yield f"✅ **Found file:** `{filepath}`\n\n```\n{file_content}\n```\n"
                        return
                    else:
                        yield f"⚠️ {file_content}\n\n"

        # ── NORMAL MODE: native tool calling (stream final answer) ────────
        memory_context = self._get_memory_context(prompt)
        messages = self._build_messages(prompt, memory_context)
        full_response = ""

        if not self.deep_reasoning_mode:
            model, provider = self.router.select_tool_capable_model(prompt, self.available_models)
            self._last_provider = provider
            self._last_model_label = {
                "llama-3.1-8b-instant":            "⚡ 8B",
                "llama-3.3-70b-versatile":          "🔥 70B",
                "qwen-3-235b-a22b-instruct-2507":   "⚡ Qwen3",
                "Meta-Llama-3.3-70B-Instruct":      "🔥 70B",
                "DeepSeek-R1":                      "🧠 DeepSeek-R1",
                "nvidia/nemotron-3-nano-30b-a3b":   "⚡ Nemotron",
                "zai-org/GLM-5.1-FP8":              "🧠 GLM-5.1 · modal",
                "z-ai/glm-5.1":                     "🧠 GLM-5.1 · openrouter",
                "z-ai/glm-5.1-together":             "🧠 GLM-5.1 · together",
                "minimaxai/minimax-m2.7":            "🔥 MiniMax",
            }.get(model, "⚡ 8B")
            used_provider = provider
            used_model = model

            providers_to_try = [(model, provider)]
            if provider != GROQ:
                providers_to_try.append(("llama-3.3-70b-versatile", GROQ))
            providers_to_try.append(("qwen-3-235b-a22b-instruct-2507", CEREBRAS))
            providers_to_try.append(("nvidia/nemotron-3-nano-30b-a3b", NVIDIA))
            providers_to_try.append(("Meta-Llama-3.3-70B-Instruct", SAMBANOVA))
            if self.openrouter_glm_client:
                providers_to_try.append(("z-ai/glm-5.1", OPENROUTER))
            if self.together_client:
                providers_to_try.append(("z-ai/glm-5.1-together", TOGETHER))
            providers_to_try.append(("llama-3.1-8b-instant", GROQ))
            stream_gen = None
            for try_model, try_provider in providers_to_try:
                try:
                    stream_gen = self._run_tool_calling_loop_stream(
                        messages=messages,
                        model=try_model,
                        provider=try_provider,
                        max_output_tokens=max_output_tokens,
                    )
                    used_provider = try_provider
                    used_model = try_model
                    break
                except Exception:
                    continue

            if stream_gen is None:
                yield "⚠️ All providers failed."
                return
            for chunk in stream_gen:
                full_response += chunk
                yield chunk
            # _last_provider and _last_model_label updated inside synthesis loop if fallback triggered
            model_label = self._last_model_label
            used_provider = self._last_provider
        # ── DEEP REASONING MODE: DeepSeek-R1 + keyword fallback ──────────
        else:
            model, provider = self.router.select_model(
                prompt, self.available_models, force_reasoning=True
            )
            model_label = self.router.get_complexity_label(prompt)
            self._last_model_label = model_label
            self._last_provider = provider
            used_provider = provider

            # Keyword trigger fallback for search
            if any(t in prompt.lower() for t in _SEARCH_TRIGGERS):
                search_context = self.tavily_search(prompt)
                if search_context:
                    memory_context = f"{search_context}\n\n{memory_context}".strip()
                    messages = self._build_messages(prompt, memory_context)

            providers_to_try = [(model, provider)]
            if provider != GROQ:
                providers_to_try.append(("llama-3.3-70b-versatile", GROQ))
            providers_to_try.append(("qwen-3-235b-a22b-instruct-2507", CEREBRAS))
            providers_to_try.append(("nvidia/nemotron-3-nano-30b-a3b", NVIDIA))
            providers_to_try.append(("Meta-Llama-3.3-70B-Instruct", SAMBANOVA))
            if self.modal_client:
                providers_to_try.append(("zai-org/GLM-5.1-FP8", MODAL))
            if self.openrouter_glm_client:
                providers_to_try.append(("z-ai/glm-5.1", OPENROUTER))
            if self.together_client:
                providers_to_try.append(("z-ai/glm-5.1-together", TOGETHER))
            providers_to_try.append(("llama-3.1-8b-instant", GROQ))
            stream = None
            for try_model, try_provider in providers_to_try:
                try:
                    stream = self._call_provider(
                        model=try_model,
                        provider=try_provider,
                        messages=messages,
                        max_tokens=max_output_tokens,
                        stream=True,
                    )
                    used_provider = try_provider
                    break
                except Exception:
                    continue

            if stream is None:
                yield "⚠️ All providers failed."
                return

            in_think_block = False
            think_content = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    text = delta.content
                    full_response += text
                    if "<think>" in text:
                        in_think_block = True
                        continue
                    if "</think>" in text:
                        in_think_block = False
                        continue
                    if in_think_block:
                        think_content += text
                    else:
                        yield text
            # If nothing was yielded outside think blocks, show the thinking
            if not full_response.replace(think_content, "").strip():
                yield think_content

        # ── POST-STREAM: code execution feedback loop ─────────────────────
        think_stripped = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
        if think_stripped:
            full_response = think_stripped
        else:
            # DeepSeek-R1 put entire answer inside <think> — extract it
            think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
            full_response = think_match.group(1).strip() if think_match else full_response

        code_matches = re.findall(r'```python\s*(.*?)```', full_response, re.DOTALL)
        if code_matches and any(w in prompt.lower() for w in [
            "write", "create", "build", "code", "script", "function",
            "implement", "calculate", "compute", "solve", "run"
        ]):
            code = "\n\n".join(block.strip() for block in code_matches)
            from src.code_runner import SafeCodeRunner
            run_result = SafeCodeRunner.run(code)

            if run_result['success'] and run_result.get('output'):
                note = f"\n\n✅ **Code executed** ({run_result['execution_time_ms']}ms):\n```\n{run_result['output']}\n```"
                full_response += note
                yield note

            elif not run_result['success']:
                error = run_result['error']
                yield f"\n\n⚠️ **Code error, fixing...**\n```\n{error}\n```\n\n"

                fix_messages = self._build_messages(
                    f"Fix this Python error:\n{error}\n\nCode:\n```python\n{code}\n```\n\nReturn only corrected code in a ```python block.",
                    ""
                )
                fix_model, fix_provider = self.router.select_tool_capable_model(prompt, self.available_models)
                try:
                    fix_stream = self._call_provider(
                        model=fix_model, provider=fix_provider,
                        messages=fix_messages, max_tokens=1024, stream=True,
                    )
                    fix_response = ""
                    for chunk in fix_stream:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            fix_response += delta.content
                            yield delta.content

                    fix_match = re.search(r'```python\s*(.*?)```', fix_response, re.DOTALL)
                    if fix_match:
                        fix_run = SafeCodeRunner.run(fix_match.group(1).strip())
                        if fix_run['success'] and fix_run.get('output'):
                            fix_note = f"\n\n✅ **Fixed & executed** ({fix_run['execution_time_ms']}ms):\n```\n{fix_run['output']}\n```"
                            full_response += fix_note
                            yield fix_note
                        elif not fix_run['success']:
                            yield f"\n\n❌ **Auto-fix failed:** `{fix_run['error'][:200]}`"
                except Exception as e:
                    yield f"\n\n❌ **Fix attempt failed:** `{str(e)[:150]}`"

        # ── FINALIZE ──────────────────────────────────────────────────────
        if self.caveman_mode:
            full_response = compress_response(full_response)

        self._append_to_history(prompt, full_response)
        self.tokens_used_today += self._count_tokens(full_response)

        yield f"\n\n`{model_label} · {used_provider}`"

    # ── UTILITY ───────────────────────────────────────────────────────────────
    def clear_history(self):
        """Clear in-session conversation history (keep long-term memory)."""
        self.conversation_history = []

    def clear_all(self):
        """Clear history + long-term memory."""
        self.conversation_history = []
        self.memory.clear()

    def get_usage(self) -> dict:
        """Get current token usage stats."""
        self._reset_daily_if_needed()
        elapsed = (datetime.now() - self.last_reset).seconds
        reset_in = max(0, 24 - elapsed // 3600)
        providers = [GROQ]
        if self.sambanova_client:
            providers.append(SAMBANOVA)
        if self.cerebras_client:
            providers.append(CEREBRAS)
        if self.nvidia_client:
            providers.append(NVIDIA)
        if self.modal_client:
            providers.append(MODAL)
        if self.openrouter_glm_client:
            providers.append(OPENROUTER)
        if self.together_client:
            providers.append(TOGETHER)
        if self.minimax_client:
            providers.append(MINIMAX)
        return {
            "tokens_used_today": self.tokens_used_today,
            "daily_limit": self.daily_token_limit,
            "remaining": max(0, self.daily_token_limit - self.tokens_used_today),
            "reset_in_hours": reset_in,
            "providers_active": providers,
            "history_turns": len(self.conversation_history) // 2,
        }
