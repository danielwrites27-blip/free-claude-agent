# 🤖 Free Claude Agent

> **A 100% free, self-hosted AI agent with persistent semantic memory, 7 real tools, and agentic reasoning.**  
> Multi-provider (Cerebras/Groq/SambaNova) • Hybrid Memory (BM25 + Vector + RRF) • ReAct Loop • Self-Aware

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by Cerebras](https://img.shields.io/badge/Powered%20by-Cerebras-blue)](https://cerebras.ai)
[![Fallback: Groq](https://img.shields.io/badge/Fallback-Groq-black?logo=groq)](https://groq.com)

**Live demo:** https://rdnqupanczhe.ap-southeast-1.clawcloudrun.com/

---

## ✨ Key Features

### 🧠 Agentic ReAct Loop
Not a chatbot — a proper agent. Uses a Plan → Tool → Observe → Synthesize loop:
- **Planning nudge** — states its plan before every tool call
- **Self-correction** — enriches error results with retry guidance and loops again
- **Loop detection** — blocks duplicate `(tool, args)` pairs via a `seen_tool_calls` set so it never spins forever
- **Three-man-team review** — lightweight self-check pass after every synthesized answer

### 🔍 SocratiCode Hybrid Memory
Memories persist across conversations in ChromaDB on a 5 GB volume. Every recall runs two searches in parallel and fuses them:

```
                        Query
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
    BM25 keyword search        Vector search
    (rank-bm25)                (ChromaDB cosine)
    ranked list A              ranked list B
              │                       │
              └───────────┬───────────┘
                          ▼
                     RRF fusion
               score = Σ 1/(60 + rank)
          (docs in both lists score higher)
                          │
                          ▼
            Re-score: × confidence × recency
            (decays 30 days, boosts on re-store)
                          │
                          ▼
            Token budget enforced → return
```

### 🛠️ 7 Real Tools

| Tool | What it does |
|---|---|
| `web_search` | Tavily search — 3 results injected as context |
| `run_python` | Sandboxed Python with auto-fix loop |
| `read_file` | Reads agent's own source files into context |
| `fetch_url` | HTTP GET with content extraction |
| `recall_memory` | Hybrid BM25 + vector memory search |
| `store_memory` | Persist a fact with confidence scoring |
| `calculate` | Safe arithmetic expression evaluator |

### 🚀 Multi-Provider Fallback
Never goes down. Routes by complexity and availability:

```
Tool calling (normal mode)
  Primary:   Cerebras  / qwen-3-235b-a22b-instruct-2507   ← fastest free inference
  Fallback1: Groq      / llama-3.1-8b-instant
  Fallback2: SambaNova / Meta-Llama-3.3-70B-Instruct
  Fallback3: SambaNova / Meta-Llama-3.1-8B-Instruct

Deep reasoning mode
  Primary:   SambaNova / DeepSeek-R1-0528
  Fallback:  Groq 70B → Cerebras Qwen3 → SambaNova Llama → Groq 8B
```

### 🦕 Caveman Mode (Token Saver)
Strips filler words ("Sure!", "I think", "Additionally") to save ~75% on output tokens. Preserves code blocks and technical accuracy.

### 🔒 Self-Aware Debugging
The agent can read its own source files to answer questions about its own behaviour — referencing actual function names, line numbers, and variable values instead of guessing.

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/danielwrites27-blip/free-claude-agent.git
cd free-claude-agent
pip install -r requirements.txt
```

### 2. Configure environment
```bash
export GROQ_API_KEY=gsk_...          # Required — fallback tool calling
export CEREBRAS_API_KEY=...          # Required — primary tool calling
export SAMBANOVA_API_KEY=...         # Required — DeepSeek-R1 deep reasoning
export TAVILY_API_KEY=...            # Required — web search (1000 free credits/month)
```

Or create a `.env` file with the same keys.

### 3. Run
```bash
python app.py
# Open http://localhost:7860
```

ChromaDB writes to `/app/data/chromadb` by default. Create that path or change it via the `MEMORY_PATH` env var.

---

## 💡 Usage Examples

**Web search:**
```
User:  "What's the gold price today?"
Agent: [calls web_search] → fetches Tavily results → synthesizes answer with sources
```

**Memory across sessions:**
```
User:  "Remember that I prefer dark mode in all apps"
Agent: [calls store_memory] → "Stored. I'll remember that."

(next session)
User:  "What are my UI preferences?"
Agent: [calls recall_memory] → "You prefer dark mode in all applications."
```

**Deep reasoning:**
```
User:  "Compare PostgreSQL vs MongoDB for a high-write chat app" (enable Deep Reasoning)
Agent: [routes to DeepSeek-R1] → step-by-step analysis with trade-offs
```

**Self-debugging:**
```
User:  "Why is memory recall returning empty results?"
Agent: [calls read_file on src/memory.py] → references actual function logic → explains the issue
```

**Python execution:**
```
User:  "Calculate the first 10 Fibonacci numbers"
Agent: [calls run_python] → executes code → returns output
```

---

## 🎛️ UI Modes

| Mode | Description | Best for |
|---|---|---|
| 🦕 Caveman | Strips filler words | Saving tokens, fast answers |
| 🧠 Deep Reasoning | Forces DeepSeek-R1 | Hard math, logic, coding |
| ⚖️ Normal | Cerebras Qwen3 with tools | General use |
| 🔒 Priority rule | Deep Reasoning overrides Caveman if both checked | Complex tasks needing precision |

---

## 🏗️ Architecture

```
free-claude-agent/
├── app.py              — Gradio UI, chat_stream(), token usage, rotating log
├── entrypoint.sh       — fixes /app/data permissions, drops to appuser via gosu
├── requirements.txt    — all dependencies
└── src/
    ├── agent.py        — FreeAgent: ReAct loop, 7 tools, planning, self-correction
    ├── caveman.py      — token compression mode
    ├── memory.py       — SocratiCode: ChromaDB + BM25 hybrid memory
    ├── router.py       — ModelRouter: complexity-based model selection
    └── code_runner.py  — SafeCodeRunner: sandboxed Python with auto-fix loop
```

### How the agentic loop works

```
User message
    │
    ▼
_build_messages()
    └── injects system prompt + recalled memories + file context
    │
    ▼
_run_tool_calling_loop_stream()
    │
    ├── [plan]      agent states plan before tool calls
    │
    ├── [tool call] one tool per round
    │
    ├── [observe]   result injected back into context
    │     └── [error?] self-correction guidance added → retry
    │
    ├── [loop check] duplicate (tool, args) blocked
    │
    └── [synthesis]
          │
          ▼
    _review_pass()   ← three-man-team self-check
          │
          ▼
    Stream to Gradio
```

---

## ☁️ Deployment (run.claw.cloud)

The agent runs on Kubernetes. `/app/` is **not persistent** — only `/app/data/` survives pod recreations (5 GB volume, ChromaDB lives here).

### The only correct deployment path

1. Edit files in GitHub UI
2. GitHub Actions rebuilds the image (2–3 min) — monitor at the [Actions tab](https://github.com/danielwrites27-blip/free-claude-agent/actions)
3. Delete app + create new app in run.claw.cloud dashboard
4. Set **Command**: `/entrypoint.sh` in Advanced Config (Arguments: empty)
5. Mount volume to `/app/data` (5 GB)

> **Never** rely on direct container edits — they are wiped on every pod recreation.

### Environment variables

| Variable | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Fallback tool calling |
| `CEREBRAS_API_KEY` | ✅ | Primary (Qwen3-235B) |
| `SAMBANOVA_API_KEY` | ✅ | DeepSeek-R1 deep reasoning |
| `TAVILY_API_KEY` | ✅ | Web search |
| `GEMINI_API_KEY` | Optional | Wired, not in active rotation |
| `OPENROUTER_API_KEY` | Optional | Wired, not in active rotation |
| `DAILY_TOKEN_LIMIT` | Optional | Default 50000 |
| `HF_AUTH_USERNAME` | Optional | Basic auth |
| `HF_AUTH_PASSWORD` | Optional | Basic auth |

### Useful pod commands

```bash
# Verify process and user
cat /proc/1/cmdline | tr '\0' ' '
cat /proc/1/status | grep Uid            # should show 1000 (appuser)

# Check memory volume
ls -la /app/data/chromadb/
python3 -c "
import chromadb
c = chromadb.PersistentClient(path='/app/data/chromadb')
col = c.get_or_create_collection('agent-memories')
print('Memories stored:', col.count())
"

# Tail logs
tail -f /app/agent.log
```

---

## 🌿 Branches

| Branch | Purpose |
|---|---|
| `main` | Active development |
| `stable-session-10` | Frozen working state before SocratiCode — safe rollback point |

---

## 📄 License

MIT — fork and modify freely.

## 🙏 Credits

Built with: [Cerebras](https://cerebras.ai) · [Groq](https://groq.com) · [SambaNova](https://sambanova.ai) · [Gradio](https://gradio.app) · [ChromaDB](https://www.trychroma.com) · [Tavily](https://tavily.com) · [rank-bm25](https://github.com/dorianbrown/rank_bm25)
