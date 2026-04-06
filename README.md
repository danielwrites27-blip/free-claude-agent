# 🆓 Free Claude Agent

A **100% free**, token-optimized AI agent with Claude-style reasoning. No credit card, no GPU management, no monthly fees.

✨ **Features**
- 🚀 **Groq free tier**: 1,000-14,400 requests/day (no card required)
- 💬 **Caveman mode**: ~75% token savings on outputs
- 🧠 **Smart routing**: Auto-select cheapest capable model (8B vs 70B)
- 💾 **SQLite memory**: Single-file conversation history, no vector DB
- 🌐 **One-click deploy**: GitHub Actions → Hugging Face Spaces

## 🚀 Quick Start

### Option 1: Try Live Demo
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/free-claude-agent)

### Option 2: Run Locally (5 minutes)
```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/free-claude-agent
cd free-claude-agent

# 2. Get free API key
# Visit: https://console.groq.com/keys

# 3. Configure environment
cp .env.example .env
# Edit .env: Add your GROQ_API_KEY

# 4. Run with Docker (recommended)
docker build -t free-agent .
docker run -p 7860:7860 --env-file .env free-agent

# Or run directly (requires Python 3.11+)
pip install -r requirements.txt
python app.py
