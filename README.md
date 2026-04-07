# 🆓 Free Claude Agent

A **100% free**, token-optimized AI agent with Claude-style reasoning.
No credit card, no GPU, no monthly fees.

✨ **Features**
- 🚀 **Multi-provider**: Groq → Cerebras → SambaNova automatic fallback
- 💬 **Caveman mode**: ~75% token savings on outputs
- 🧠 **Deep reasoning**: Chain-of-thought for complex problems
- 💾 **SQLite memory**: Conversation history persists across sessions
- 📊 **Metrics tracking**: Monitor token usage and performance
- 🐳 **Docker deploy**: One command to run anywhere

## 🚀 Quick Start

### Option 1: Try Live Demo
👉 https://htiicjduzjbi.ap-southeast-1.clawcloudrun.com

### Option 2: Self-host with Docker
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
```

Then open http://localhost:7860

## 🔑 API Keys (all free)

| Provider | Free Limit | Get Key |
|---|---|---|
| Groq | 1,000 req/day (70B) | https://console.groq.com/keys |
| Cerebras | 1M tokens/day | https://cloud.cerebras.ai |
| SambaNova | $5 free credit | https://cloud.sambanova.ai |

## 💡 Tips
- **Simple questions**: Leave Deep Reasoning off for fastest responses
- **Complex problems**: Enable 🧠 Deep Reasoning for step-by-step analysis
- **Memory**: Agent remembers past conversations automatically
- **Reset**: Use the Reset Memory button to start fresh

## 🏗️ Architecture
