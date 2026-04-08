# 🆓 Free Claude Agent

A 100% free, token-optimized AI agent with **4 distinct personalities**, self-debugging capabilities, and automatic multi-provider fallback. No credit card, no GPU, no monthly fees.

## ✨ New Features (v2.0)

- **🎭 4 Operating Modes**: Switch between Caveman, Deep Reasoning, Normal, and Priority modes instantly
- **📂 Self-Debugging**: Agent can read its own source code (`app.py`, `src/agent.py`) to fix errors
- **🌐 Live URL Checking**: Fetches and summarizes web pages in real-time
- **🧠 Deep Reasoning**: Step-by-step chain-of-thought for complex coding & math
- **🦕 Caveman Mode**: Strips filler words for ~75% token savings
- **💾 Smart Memory**: SQLite persistence with BM25 recall across sessions
- **🔄 Auto-Failover**: Groq → Sambanova → Cerebras (never goes down)

---

## 🎭 The 4 Modes

| Mode | Caveman | Deep Reasoning | Personality | Best For |
|------|---------|----------------|-------------|----------|
| **🦕 Caveman** | ✅ ON | ❌ OFF | "Ugga see error. Fix line 10." | Quick facts, saving tokens |
| **🧠 Deep Reasoning** | ❌ OFF | ✅ ON | "Let's analyze step-by-step..." | Complex coding, math, learning |
| **⚖️ Normal** | ❌ OFF | ❌ OFF | "Sure! I can help with that..." | Casual chat, general questions |
| **🚀 Deep Priority** | ✅ ON | ✅ ON | Same as Deep Reasoning | When you want max intelligence |

> **Note:** If both boxes are checked, **Deep Reasoning wins** (intelligence > brevity).

---

## 🚀 Quick Start

### Self-host with Docker

```bash
# 1. Clone repo
git clone https://github.com/danielwrites27-blip/free-claude-agent 
cd free-claude-agent

# 2. Get free API keys
# Groq:       https://console.groq.com/keys 
# Cerebras:   https://cloud.cerebras.ai 
# SambaNova:  https://cloud.sambanova.ai 

# 3. Configure environment
cp .env.example .env
# Edit .env: add your API keys

# 4. Run
docker build -t free-agent .
docker run -p 7860:7860 --env-file .env free-agent
Then open http://localhost:7860

🔑 API Keys (all free)
Provider                Free Limit                Get Key
Groq                    1,000 req/day (70B)       -
Cerebras                1M tokens/day             -
SambaNova               $5 free credit            -

💡 Usage Examples
1. Debug Your Own Code
User: "I have an error in app.py. Please read the file and tell me what's wrong on line 90."
Agent: Reads file automatically → "✅ Found app.py. Line 90 has a missing colon after the if statement."
2. Check Live Websites
User: "Check https://www.python.org and summarize the main news."
Agent: Fetches URL → "Ugga see Python site. News about Python 3.12 release, security updates."
3. Complex Coding Task (Deep Mode)
User: (Enables Deep Reasoning) "Write a secure authentication system with JWT and refresh tokens."
Agent: Thinks step-by-step → Provides full architecture, code, security notes, and edge cases.
4. Quick Question (Caveman Mode)
User: (Enables Caveman) "How to fix React re-render bug?"
Agent: "Inline object prop = new ref = re-render. Wrap in useMemo."

🏗️ Architecture
app.py              → Gradio UI + Mode Toggles
src/agent.py        → Core agent + File Read

📦 Stack
Python 3.11
Gradio 4.40+ (with custom toggles)
Groq SDK (Primary provider)
OpenAI SDK (For Cerebras + SambaNova compatibility)
SQLite FTS5 (Full-text search memory)
BeautifulSoup4 (URL content extraction)
Docker (Containerized deployment)

🔧 Advanced Configuration
Environment Variables (.env)
GROQ_API_KEY=your_key_here
SAMBANOVA_API_KEY=optional_key
CEREBRAS_API_KEY=optional_key
DAILY_TOKEN_LIMIT=50000
MEMORY_PATH=agent_memory.db
CAVEMAN_MODE=true       # Default mode
HF_AUTH_USERNAME=opt    # For HuggingFace Spaces auth
HF_AUTH_PASSWORD=opt

Customizing Modes
Edit src/agent.py → _build_messages() to change system prompts for each mode.

🛠️ Troubleshooting
"File not found" error:
Ensure files are committed to Git and included in Docker build
Check pod logs for exact path calculation
"All providers failed":
Verify API keys in .env
Check rate limits on provider dashboards
Agent gives generic answers:
Enable Deep Reasoning for complex tasks
Disable Caveman Mode for full English responses

📄 License
MIT License - Free for personal and commercial use.
🙏 Credits
Built with ❤️ by Daniel Writes
Powered by Groq, Cerebras, and SambaNova free tiers.
