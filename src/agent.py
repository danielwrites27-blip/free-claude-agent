"""
Token-optimized, free-tier AI agent.
Multi-provider: Groq + Sambanova + Cerebras (all free tier).
Multi-turn conversation history, streaming, SQLite memory.
"""
import os
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Generator

from groq import Groq
import tiktoken

from .caveman import CAVEMAN_SYSTEM_PROMPT, compress_response
from .memory import TokenEfficientMemory
from .router import ModelRouter, GROQ, SAMBANOVA, CEREBRAS

# Max turns to keep in live conversation context (before summarizing to memory)
MAX_HISTORY_TURNS = 12  # 12 pairs = 24 messages

class FreeAgent:
    """100% free, token-optimized reasoning agent.
    
    Supports Groq, Sambanova, Cerebras — falls back automatically.
    Maintains real multi-turn conversation history within session.
    Persists long-term memory to SQLite across sessions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        daily_token_limit: int = 50000,
        memory_path: str = "agent_memory.db",
        caveman_mode: bool = True,
        deep_reasoning_mode: bool = False,  # NEW: Added deep reasoning toggle
    ):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.caveman_mode = caveman_mode
        self.deep_reasoning_mode = deep_reasoning_mode  # NEW: Store the setting

        # ── Provider clients ──────────────────────────────────────────────
        # Groq (primary)
        groq_key = api_key or os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError(
                "GROQ_API_KEY required. Get free key: https://console.groq.com/keys  "
            )
        self.groq_client = Groq(api_key=groq_key)

        # Sambanova (secondary — generous free tier)
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
                
        # Cerebras (tertiary — fastest inference)
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
# Gemini (SVG primary — via OpenAI-compatible endpoint)
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

        # OpenRouter (SVG fallback — Qwen3.5 free tier)
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

        # Tavily (web search)
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        
        # ── Budget tracking ───────────────────────────────────────────────
        self.daily_token_limit = daily_token_limit
        self.tokens_used_today = 0
        self.last_reset = datetime.now()

        # ── Components ────────────────────────────────────────────────────
        self.memory = TokenEfficientMemory(memory_path)
        self.router = ModelRouter()

        # Multi-turn conversation history
        self.conversation_history: List[Dict] = []

        # ── Available models ──────────────────────────────────────────────
        self.available_models: Dict[str, dict] = {}
        self._register_available_models()

       # ── FILE READING METHOD ──────────────────────────────────────────────
    def read_file(self, filepath: str, line_number: int = None) -> str:
        """Reads a file from the project directory. 
        If line_number is provided, highlights that specific line."""
        try:
            current_file = os.path.abspath(__file__)
            base_dir = os.path.dirname(os.path.dirname(current_file))
            full_path = os.path.join(base_dir, filepath)
            
            if not os.path.exists(full_path):
                return f"Error: File '{filepath}' not found."
                
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # If no line number requested, return full content
            if line_number is None:
                return "".join(lines)
            
            # Validate line number
            if line_number < 1 or line_number > len(lines):
                return f"Error: Line {line_number} out of range. File has {len(lines)} lines."
            
            # Highlight specific line with context (5 lines before/after)
            start = max(0, line_number - 6)
            end = min(len(lines), line_number + 5)
            
            output = [f"📄 Showing lines {start+1}-{end} of {filepath}:\n\n```python\n"]
            
            for i in range(start, end):
                line_num = i + 1
                content = lines[i].rstrip()
                
                if line_num == line_number:
                    # Highlight the target line
                    output.append(f"{line_num:4d} >>> {content}\n")
                else:
                    output.append(f"{line_num:4d}     {content}\n")
            
            output.append("```")
            return "".join(output)
            
        except Exception as e:
            return f"Error reading file: {str(e)}"
    # ──────────────────────────────────────────────────────────────────────

            # ── SAFE FILE EDITING METHOD ──────────────────────────────────────────────
    def edit_file(self, filepath: str, new_content: str) -> str:
        """Safely edits a file with backup and validation."""
        try:
            # 1. Security Check: Prevent directory traversal and system files
            if ".." in filepath or filepath.startswith("/"):
                return "⛔ Security Error: Invalid file path."
            
            allowed_extensions = ['.py', '.txt', '.md', '.json', '.yaml', '.env']
            if not any(filepath.endswith(ext) for ext in allowed_extensions):
                return f"⛔ Security Error: Cannot edit {filepath}. Allowed: {', '.join(allowed_extensions)}"

            # 2. Calculate Path
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(base_dir, filepath)

            if not os.path.exists(full_path):
                return f"Error: File '{filepath}' not found."

            # 3. Create Backup
            backup_path = full_path + ".bak"
            with open(full_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)

            # 4. ⚠️ NEW: Validate Python Syntax BEFORE writing
            if filepath.endswith('.py'):
                try:
                    compile(new_content, full_path, 'exec')
                except SyntaxError as e:
                    # Restore backup immediately if validation fails
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                    # Remove the failed backup file to avoid confusion
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    return f"⛔ **Syntax Error Detected!** Changes rejected.\nLine {e.lineno}: {e.msg}\n💾 Backup restored automatically."

            # 5. Write New Content (Only if validation passes)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return f"✅ Successfully updated `{filepath}`.\n💾 Backup saved to `{filepath}.bak`\n🛡️ Syntax validation passed."
        
        except Exception as e:
            return f"❌ Error editing file: {str(e)}"
    # ──────────────────────────────────────────────────────────────────────────

    def _register_available_models(self):
        """Register which provider+model combos we actually have keys for"""
        # Groq always available (required key)
        self.available_models["llama-3.1-8b-instant"] = {"provider": GROQ, "rpd": 14400}
        self.available_models["llama-3.3-70b-versatile"] = {"provider": GROQ, "rpd": 1000}

        if self.sambanova_client:
            self.available_models["Meta-Llama-3.1-8B-Instruct"] = {"provider": SAMBANOVA, "rpd": 10000}
            self.available_models["Meta-Llama-3.3-70B-Instruct"] = {"provider": SAMBANOVA, "rpd": 5000}

        if self.cerebras_client:
            self.available_models["llama3.1-8b"] = {"provider": CEREBRAS, "rpd": 50000}
            self.available_models["llama-3.3-70b"] = {"provider": CEREBRAS, "rpd": 10000}

    def _reset_daily_if_needed(self):
        """Reset token counter if 24h passed"""
        now = datetime.now()
        if now - self.last_reset > timedelta(hours=24):
            self.tokens_used_today = 0
            self.last_reset = now

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

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

    def _get_memory_context(self, prompt: str) -> str:
        """Retrieve relevant long-term memories for this prompt"""
        recalled = self.memory.recall(prompt, max_tokens=800)
        
        # Check for URL in prompt and fetch content
        url_context = ""
        if "http://" in prompt or "https://" in prompt:
            urls = re.findall(r'https?://[^\s<>"]+', prompt)
            if urls:
                url = urls[0]
                fetched_content = self._fetch_url_content(url)
                if not fetched_content.startswith("Error"):
                    url_context = f"\n\nContent from {url}:\n{fetched_content}"
        
        return recalled + url_context

    def tavily_search(self, query: str) -> str:
        """Search the web via Tavily and return formatted results."""
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
                parts.append(f"{i}. {res.get('title','')}\n{res.get('url','')}\n{res.get('content','')[:900]}")
            return "\n\n".join(parts)
        except Exception as e:
            print(f"[Tavily] Search failed: {e}", flush=True)
            return ""
    
    def _extract_function(self, source: str, func_name: str) -> str:
        """Extract a specific function/method from Python source by name.
        Returns the full function body, or the first 2000 chars if not found."""
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
                # End of function: new def/class at same or lower indent level
                if (stripped and indent <= func_indent
                        and (stripped.startswith("def ")
                             or stripped.startswith("class ")
                             or stripped.startswith("async def "))):
                    break
                result.append(line)

        if result:
            return "\n".join(result)
        # Function not found — fall back to first 2000 chars of the file
        return source[:2000] + "\n... [function not found — showing file start]"

    def _get_multi_file_context(self, prompt: str) -> str:
        """
        Smartly injects relevant code snippets into the LLM context.

        - Trigger detection is broad (includes 'why', 'how', 'explain', etc.)
        - Focuses on the specific file/function mentioned in the prompt
        - Logs trigger/file/size info to stdout for debugging
        """
        prompt_lower = prompt.lower()

        # ── Expanded trigger words ────────────────────────────────────────
        # OLD list was too narrow — 'why is caveman mode not saving tokens?'
        # matched NOTHING and returned "" immediately.
        trigger_words = [
            "bug", "error", "fix", "broken", "not working", "issue", "debug",
            "feature", "add", "update",
            # Analysis words (the main missing category)
            "why", "how", "explain", "analyze", "analyse", "understand",
            "not saving", "not working", "isn't", "doesn't",
            # Component words
            "caveman", "token", "compress", "mode", "routing", "router",
            "memory", "stream", "reasoning", "context", "inject",
            "code_runner", "code runner", "code runner.py",
            "what does", "what is", "what are", "how does", "tell me",
        ]

        triggered = any(word in prompt_lower for word in trigger_words)
        print(f"[MultiFileContext] Trigger detected: {triggered}", flush=True)

        if not triggered:
            return ""

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # ── Smart focus: keyword → (file, function_to_extract) ───────────
        # Instead of dumping whole files truncated at the TOP (which misses all
        # relevant functions), we extract only the relevant function when we can.
        keyword_focus = [
            # (keyword_in_prompt, filepath, function_name_or_None)
            ("caveman",         "src/caveman.py",   None),           # small file, read whole
            ("compress",        "src/caveman.py",   None),
            ("token",           "src/caveman.py",   None),
            ("_build_messages", "src/agent.py",     "_build_messages"),
            ("deep reasoning",  "src/agent.py",     "_build_messages"),
            ("multi_file",      "src/agent.py",     "_get_multi_file_context"),
            ("context inject",  "src/agent.py",     "_get_multi_file_context"),
            ("routing",         "src/router.py",    None),
            ("router",          "src/router.py",    None),
            ("memory",          "src/memory.py",    None),
            ("stream",          "src/agent.py",     "ask_stream"),
            ("code_runner",     "src/code_runner.py", None),
            ("code runner",     "src/code_runner.py", None),
        ]

        files_to_read: dict[str, str | None] = {}  # filepath → func_name or None

        for keyword, filepath, func_name in keyword_focus:
            if keyword in prompt_lower:
                # Don't overwrite a focused extraction with a broader None
                if filepath not in files_to_read or files_to_read[filepath] is None:
                    files_to_read[filepath] = func_name

        # Default: read all three core files when no specific focus matched
        if not files_to_read:
            files_to_read = {
                "app.py":          None,
                "src/agent.py":    None,
                "src/caveman.py":  None,
            }

        print(f"[MultiFileContext] Files to read: {list(files_to_read.keys())}", flush=True)

        context_parts = [
            "--- ACTUAL PROJECT SOURCE CODE ---\n"
            "The code below is the real, current source. You MUST:\n"
            "- Reference specific function names, variable names, and exact values you can see (e.g. 'your fillers list has 12 patterns')\n"
            "- Quote specific lines when relevant (e.g. 'line X does Y')\n"
            "- Never give generic advice that ignores what is actually written here\n"
            "- If the code already handles something the user thinks is broken, say so explicitly\n"
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
                    # Smart extraction: pull only the relevant function
                    content = self._extract_function(raw, func_name)
                    label = f"{filepath} → {func_name}()"
                else:
                    # Whole file — 9000 chars gives ~750+ tokens, 2× the old limit
                    # and actually reaches the functions that matter
                    MAX_CHARS = 9000
                    # agent.py is 35k chars — too large for flat read.
                    # Extract the 3 most important functions instead.
                    if filepath == "src/agent.py" and len(raw) > MAX_CHARS:
                        key_funcs = ["ask_stream", "_build_messages", "_get_multi_file_context"]
                        parts = [self._extract_function(raw, f) for f in key_funcs]
                        content = "\n\n".join(p for p in parts if p)
                        label = f"{filepath} → [ask_stream, _build_messages, _get_multi_file_context]"
                    elif len(raw) <= MAX_CHARS:
                        content = raw
                    else:
                        content = raw[:MAX_CHARS] + f"\n... [truncated at {MAX_CHARS} chars — ask to read specific lines or functions]"
                    label = filepath

                context_parts.append(f"\n### {label}\n```python\n{content}\n```")
                total_chars += len(content)

            except Exception as e:
                print(f"[MultiFileContext] Error reading {filepath}: {e}", flush=True)
                continue

        print(f"[MultiFileContext] Total context chars injected: {total_chars}", flush=True)

        context_parts.append("--- END PROJECT CONTEXT ---")
        return "\n".join(context_parts) if len(context_parts) > 2 else ""
    
    def _build_messages(self, prompt: str, memory_context: str) -> List[Dict]:
        """
        Build the full messages array for the API call.
        Dynamically selects system prompt based on mode.
        """
        # 1. Determine System Prompt
        if self.caveman_mode:
            from .caveman import CAVEMAN_SYSTEM_PROMPT
            system_content = CAVEMAN_SYSTEM_PROMPT
        elif self.deep_reasoning_mode:
            # Deep Reasoning Mode: Forces step-by-step thinking
            system_content = (
                "You are an expert AI assistant. You ALWAYS think step-by-step before answering.\n"
                "For every query, you must analyze constraints, edge cases, and logic internally.\n"
                "Provide clear, structured, and thorough answers. Use Markdown for formatting.\n"
                "If the user asks for code, explain the logic first, then provide the solution.\n"
                "Prioritize accuracy and depth over brevity.\n"
                "When web search results are provided at the start of context, summarize the actual content naturally. Never list URLs or name source websites unless the user explicitly asks for sources or links — just answer directly using the information found as if you already know it."
            )
        else:
            # Normal Mode: Balanced helpful assistant
            system_content = (
                "You are a helpful, harmless, and honest AI assistant. "
                "Answer clearly and concisely, but provide detail when needed. "
                "When web search results are provided at the start of context, summarize the actual content naturally. Never list URLs or name source websites unless the user explicitly asks for sources or links — just answer directly using the information found as if you already know it."
            )

        # Initialize messages list ONCE
        messages = [{"role": "system", "content": system_content}]

        # 2. Inject Multi-File Context (NEW)
        multi_file_context = self._get_multi_file_context(prompt)
        if multi_file_context:
            messages.append({
                "role": "system",
                "content": multi_file_context
            })

        # 3. Inject long-term memory
        if memory_context:
            messages.append({
                "role": "system",
                "content": f"Relevant past context:\n{memory_context}"
            })

        # 4. Rolling conversation history
        history_to_include = self.conversation_history[-MAX_HISTORY_TURNS * 2:]
        messages.extend(history_to_include)

        # 5. Current user message
        messages.append({"role": "user", "content": prompt})

        return messages

    def _call_provider(
        self,
        model: str,
        provider: str,
        messages: List[Dict],
        max_tokens: int,
        stream: bool = False,
    ):
        """Dispatch API call to the correct provider client"""
        kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.95,
            stream=stream,
        )

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

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def ask(self, prompt: str, max_output_tokens: int = 1024) -> str:
        """
        Main entry point: Ask the agent a question.
        Returns full response string (non-streaming).
        """
        self._reset_daily_if_needed()

        if self.tokens_used_today >= self.daily_token_limit:
            return "⚠️ Daily token limit reached. Resets in 24h."

        memory_context = self._get_memory_context(prompt)
        # Web search injection
        search_triggers = ["search for", "look up", "find me", "what is the latest",
                           "latest", "current", "news about", "who is", "what happened",
                           "today", "recently", "right now"]
        search_context = ""
        if any(t in prompt.lower() for t in search_triggers):
            search_context = self.tavily_search(prompt)
            if search_context:
                memory_context = f"{search_context}\n\n{memory_context}".strip()

        messages = self._build_messages(prompt, memory_context)
        model, provider = self.router.select_model(prompt, self.available_models, force_reasoning=self.deep_reasoning_mode)
        model_label = "🧠 70B" if self.deep_reasoning_mode else self.router.get_complexity_label(prompt)
        self._last_model_label = model_label
        self._last_provider = provider
        # Try selected provider, fall back to Groq if it fails
        providers_to_try = [(model, provider)]
        if provider != GROQ:
            providers_to_try.append(("llama-3.3-70b-versatile", GROQ))
        providers_to_try.append(("llama-3.1-8b-instant", GROQ))  # ultimate fallback

        raw_output = None
        tokens_used = 0
        used_model = model
        used_provider = provider

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
                tokens_used = getattr(response.usage, "total_tokens", 0)
                used_model = try_model
                used_provider = try_provider
                break
            except Exception as e:
                last_error = str(e)
                continue

        if raw_output is None:
            return f"⚠️ All providers failed. Last error: {last_error[:150]}"

        # Compress output if caveman mode
        output = compress_response(raw_output) if self.caveman_mode else raw_output

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": output})

        # Trim history if too long (keep last MAX_HISTORY_TURNS pairs)
        if len(self.conversation_history) > MAX_HISTORY_TURNS * 2:
            # Summarize and store oldest turns to long-term memory before dropping
            dropped = self.conversation_history[:2]
            summary = f"Q: {dropped[0]['content']}\nA: {dropped[1]['content']}"
            self.memory.store(content=summary, tags=["conversation"], token_count=None)
            self.conversation_history = self.conversation_history[2:]

        # Track token usage
        self.tokens_used_today += tokens_used

        # Append model label footer
        output += f"\n\n`{model_label} · {used_provider}`"

        return output

    def ask_stream(self, prompt: str, max_output_tokens: int = 1024) -> Generator[str, None, None]:
        """
        Streaming version of ask(). Yields chunks as they arrive.
        Finalizes history + memory after stream completes.
        Use with Gradio's streaming support.
        """
        self._reset_daily_if_needed()

        if self.tokens_used_today >= self.daily_token_limit:
            yield "⚠️ Daily token limit reached. Resets in 24h."
            return
# ── FILE READING INTERCEPTOR ───────────────────────────────────────────
        # Detect if user wants ONLY to read/show a file (no analysis needed)
        # If the prompt also asks to "fix", "analyze", "why", "bug", we let it pass to the AI
        simple_read_only = any(word in prompt.lower() for word in ["show me ", "check line", "what is on line", "line "])
        # Keep in sync with _get_multi_file_context trigger_words so any prompt
        # that would inject file context is also protected from interceptor early-return
        has_analysis_request = any(word in prompt.lower() for word in [
            "fix", "analyze", "analyse", "why", "how", "bug", "error",
            "tell me why", "explain", "understand", "caveman", "token",
            "compress", "mode", "routing", "router", "memory", "stream",
            "reasoning", "context", "not saving", "isn't", "doesn't",
        ])
        
        if simple_read_only and not has_analysis_request:
            # Extract filename
            match = re.search(r'([\w\./]+\.(py|txt|md|json|yaml|yml|env|toml))', prompt)
            if match:
                filepath = match.group(1)
                # Only add src/ if the file is known to be in src/
                known_src_files = ['agent.py', 'caveman.py', 'memory.py', 'router.py']
                if "/" not in filepath and filepath in known_src_files:
                    filepath = "src/" + filepath
                
                # Check if user asked for a specific line number
                line_match = re.search(r'line\s*(\d+)', prompt.lower())
                target_line = int(line_match.group(1)) if line_match else None
                
                file_content = self.read_file(filepath, line_number=target_line)
                
                if not file_content.startswith("Error"):
                    if target_line:
                        yield f"✅ **Line {target_line} of `{filepath}`:**\n\n"
                    else:
                        yield f"✅ **Found file:** `{filepath}`\n\n"
                    yield f"{file_content}\n"
                    return # Stop here ONLY for simple reads
                else:
                    yield f"⚠️ {file_content}\n\n"
                    # Continue to AI to explain the error
        # ──────────────────────────────────────────────────────────────────────

        # ── FILE OPERATIONS INTERCEPTOR ───────────────────────────────────────────────
        lower_prompt = prompt.lower()
        
        # Detect File Operations
        if any(word in lower_prompt for word in ["read file", "read ", "show me ", "check line", "what is on line", "edit ", "update ", "change "]):
            
            # Extract filename
            match = re.search(r'([\w\./]+\.(py|txt|md|json|yaml|yml|env|toml))', lower_prompt)
            if match:
                filepath = match.group(1)
                
                # Only add src/ if the file is known to be in src/
                known_src_files = ['agent.py', 'caveman.py', 'memory.py', 'router.py']
                if "/" not in filepath and filepath in known_src_files:
                    filepath = "src/" + filepath
                
                # Check for Line Number Request
                line_match = re.search(r'line\s*(\d+)', lower_prompt)
                if line_match and ("read" in lower_prompt or "show" in lower_prompt or "check" in lower_prompt):
                    line_num = int(line_match.group(1))
                    result = self.read_file(filepath, line_number=line_num)
                    yield f"{result}\n"
                    return

                # Check for Edit Request with Full Content
                if any(word in lower_prompt for word in ["edit", "update", "change", "replace"]):
                    # Look for code block with new content
                    code_match = re.search(r'```(?:python)?\s*(.*?)```', prompt, re.DOTALL)
                    
                    if code_match:
                        # User provided full new content in code block
                        new_content = code_match.group(1).strip()
                        result = self.edit_file(filepath, new_content)
                        yield f"{result}\n"
                        return
                    else:
                        # No code block found, ask for it
                        yield f"🛠️ I can edit `{filepath}` safely (with backup).\nPlease provide the **full new content** in a code block like this:\n\n```python\n# your new code here\n```\n"
                        return

                # Default: Read File (BUT check if we should stop or continue to AI)
                # If the user asked to "fix", "analyze", "why", "bug", DO NOT return here.
                # Let the code fall through to the AI so it can analyze the content.
                if not has_analysis_request: 
                    file_content = self.read_file(filepath)
                    if not file_content.startswith("Error"):
                        yield f"✅ **Found file:** `{filepath}`\n\n```\n{file_content}\n```\n"
                        return
                    else:
                        yield f"⚠️ {file_content}\n\n"
                # If has_analysis_request is True, we skip the return above 
                # and let the code continue to _build_messages below.
        # ──────────────────────────────────────────────────────────────────────

        memory_context = self._get_memory_context(prompt)
        search_triggers = ["search for", "look up", "find me", "what is the latest",
                           "latest", "current", "news about", "who is", "what happened",
                           "today", "recently", "right now"]
        if any(t in prompt.lower() for t in search_triggers):
            search_context = self.tavily_search(prompt)
            if search_context:
                memory_context = f"{search_context}\n\n{memory_context}".strip()
        messages = self._build_messages(prompt, memory_context)

        model, provider = self.router.select_model(prompt, self.available_models, force_reasoning=self.deep_reasoning_mode)
        model_label = "🧠 70B" if self.deep_reasoning_mode else self.router.get_complexity_label(prompt)
        self._last_model_label = model_label
        self._last_provider = provider
        # Try selected provider, fall back gracefully
        providers_to_try = [(model, provider)]
        if provider != GROQ:
            providers_to_try.append(("llama-3.3-70b-versatile", GROQ))
        providers_to_try.append(("llama-3.1-8b-instant", GROQ))

        stream = None
        used_provider = provider

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

        # Stream chunks
        full_response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                text = delta.content
                full_response += text
                yield text

        # Post-stream: compress, store, update history
        if self.caveman_mode:
            full_response = compress_response(full_response)

        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": full_response})

        if len(self.conversation_history) > MAX_HISTORY_TURNS * 2:
            dropped = self.conversation_history[:2]
            summary = f"Q: {dropped[0]['content']}\nA: {dropped[1]['content']}"
            self.memory.store(content=summary, tags=["conversation"], token_count=None)
            self.conversation_history = self.conversation_history[2:]

        token_estimate = self._count_tokens(full_response)
        self.tokens_used_today += token_estimate

        yield f"\n\n`{model_label} · {used_provider}`"

    def clear_history(self):
        """Clear in-session conversation history (keep long-term memory)"""
        self.conversation_history = []

    def clear_all(self):
        """Nuclear option: clear history + long-term memory"""
        self.conversation_history = []
        self.memory.clear()

    def get_usage(self) -> dict:
        """Get current token usage stats"""
        self._reset_daily_if_needed()
        elapsed = (datetime.now() - self.last_reset).seconds
        reset_in = max(0, 24 - elapsed // 3600)
        providers = [GROQ]
        if self.sambanova_client:
            providers.append(SAMBANOVA)
        if self.cerebras_client:
            providers.append(CEREBRAS)

        return {
            "tokens_used_today": self.tokens_used_today,
            "daily_limit": self.daily_token_limit,
            "remaining": max(0, self.daily_token_limit - self.tokens_used_today),
            "reset_in_hours": reset_in,
            "providers_active": providers,
            "history_turns": len(self.conversation_history) // 2,
        }
