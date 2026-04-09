# 🆓 Free Claude Agent

> **A 100% free, self-hosted AI agent that can read, analyze, and fix its own source code.**  
> Multi-provider (Groq/SambaNova/Cerebras) • Token-optimized • SQLite Memory • Self-Editing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-black?logo=groq)](https://groq.com)

## ✨ Key Features

### 🧠 Self-Aware Debugging
Unlike standard chatbots, this agent can **read its own source code** to answer questions about its behavior.
- **Multi-File Context:** Automatically injects relevant code snippets (`app.py`, `src/agent.py`, `src/caveman.py`) when you ask "Why isn't X working?" or "Fix the bug in...".
- **Evidence-Based Answers:** Instead of generic advice, it references specific lines, variables, and function logic from your actual codebase.
- **Smart Triggers:** Detects analysis keywords ("why", "how", "bug", "broken") to switch from chat mode to debug mode.

### 🛠️ Safe Self-Editing
The agent can modify its own files to apply fixes instantly.
- **Syntax Validation:** Uses Python's `compile()` to verify code correctness *before* saving. Broken code is automatically rejected.
- **Automatic Backups:** Creates a `.bak` file before every edit.
- **Security:** Prevents directory traversal and restricts edits to safe extensions (`.py`, `.json`, `.md`).

### 🦕 Caveman Mode (Token Saver)
Strip filler words ("Sure!", "I think", "Additionally") to save **~75% on output tokens**.
- Perfect for high-volume usage on free tiers.
- Preserves code blocks and technical accuracy.

### 🚀 Multi-Provider Fallback
Never go down. Automatically routes requests based on complexity and availability:
1.  **Groq** (Primary - Fastest)
2.  **SambaNova** (Secondary - High Quota)
3.  **Cerebras** (Tertiary - Backup)

### 💾 Persistent Memory
- **Short-Term:** Keeps last 12 conversation turns in live context.
- **Long-Term:** Summarizes old turns and stores them in **SQLite** (`agent_memory.db`) for retrieval across sessions.

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/free-claude-agent.git
cd free-claude-agent
pip install -r requirements.txt

2. Configure Environment
Create a .env file in the root directory:
# Required: Get free key at https://console.groq.com
GROQ_API_KEY=gsk_...

# Optional: Extend limits with free keys from SambaNova/Cerebras
SAMBANOVA_API_KEY=...
CEREBRAS_API_KEY=...

# Optional: Limits
DAILY_TOKEN_LIMIT=50000
MEMORY_PATH=agent_memory.db

3. Run Locally
python app.py
Open http://localhost:7860 in your browser.

4. Deploy to Cloud (ClawCloud / HuggingFace)
The project includes a Dockerfile for easy deployment.
docker build -t free-agent .
docker run -p 7860:7860 --env-file .env free-agent

💡 Usage Examples
1. Debugging Self
User: "Why is caveman mode not saving tokens?"
Agent: "I see in src/caveman.py line 15 that your fillers list only has 12 patterns. It misses common words like 'actually'. Also, line 42 skips articles inside code blocks..."
2. Reading Code
User: "Show me line 100 of app.py"
Agent: (Instantly displays lines 95-105 with line 100 highlighted, no API call used)
3. Editing Code
User: "Edit src/caveman.py to add 'basically' to the fillers list."
Agent: (Validates syntax, creates backup, applies fix)
"✅ Successfully updated src/caveman.py. Backup saved. Syntax validation passed."
4. Complex Reasoning
User: "Compare PostgreSQL vs MongoDB for a chat app" (Enable Deep Reasoning toggle)
Agent: (Uses 70B model, step-by-step analysis, structured output)

🎛️ Modes
Mode                    Description                                        Best For
🦕 Caveman              Strips filler words/articles                       Saving tokens, fast answers
🧠 Deep Reasoning       Forces step-by-step Chain-of-Thought     	       Coding, Math, Logic puzzles
⚖️ Normal               Balanced helpful assistant                         General chat
🔒 Priority Logic       Deep Reasoning overrides Caveman if both checked   Complex tasks requiring precision

🏗️ Architecture

free-claude-agent/
├── app.py              # Gradio UI & Singleton Agent Manager
├── src/
│   ├── agent.py        # Core Logic, Interceptors, Multi-File Context
│   ├── caveman.py      # Token Compression Logic
│   ├── memory.py       # SQLite Long-Term Memory
│   └── router.py       # Smart Model Selection (8B vs 70B)
├── agent_memory.db     # Persistent SQLite Database
└── requirements.txt    # Dependencies

How Self-Debugging Works
Intercept: User asks "Why is X broken?".
Trigger: _get_multi_file_context detects keywords ("broken", "why").
Extract: Agent reads relevant files (src/agent.py, src/caveman.py).
Inject: Code snippets are injected into the system prompt.
Analyze: LLM receives real code + question → generates specific fix.

📄 License
MIT License - feel free to fork and modify!
🙏 Credits
Built with:
Groq for blazing fast inference.
Gradio for the UI.
SambaNova & Cerebras for fallback capacity.


