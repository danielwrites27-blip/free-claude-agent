"""
Gradio interface for Free Claude Agent.
Run locally:   python app.py
Deploy:        GitHub Actions auto-pushes to HF Spaces / claw.cloud pod
"""
import os
import gradio as gr
from src.agent import FreeAgent

# ── Agent singleton (lazy-loaded on first request) ────────────────────────────
_agent: FreeAgent | None = None

def get_agent() -> FreeAgent:
    global _agent
    if _agent is None:
        _agent = FreeAgent(
            api_key=os.getenv("GROQ_API_KEY"),
            daily_token_limit=int(os.getenv("DAILY_TOKEN_LIMIT", "50000")),
            memory_path=os.getenv("MEMORY_PATH", "agent_memory.db"),  # FIXED: was .mv2
            caveman_mode=os.getenv("CAVEMAN_MODE", "true").lower() == "true",
        )
    return _agent


# ── Chat handler (streaming) ──────────────────────────────────────────────────
def chat_stream(message: str, history: list):
    """
    Gradio streaming chat handler.
    `history` is passed by Gradio automatically — we don't need it ourselves
    because the agent maintains its own conversation_history internally.
    """
    agent = get_agent()
    partial = ""
    try:
        for chunk in agent.ask_stream(message):
            partial += chunk
            yield partial
    except Exception as e:
        yield f"⚠️ Error: {str(e)[:300]}"


# ── Usage stats ───────────────────────────────────────────────────────────────
def show_usage() -> str:
    usage = get_agent().get_usage()
    providers = ", ".join(usage["providers_active"])
    return (
        f"**📊 Token Usage**\n\n"
        f"- Used today: `{usage['tokens_used_today']:,}` / `{usage['daily_limit']:,}`\n"
        f"- Remaining: `{usage['remaining']:,}`\n"
        f"- Resets in: ~`{usage['reset_in_hours']}h`\n"
        f"- Active providers: `{providers}`\n"
        f"- Conversation turns: `{usage['history_turns']}`"
    )


# ── Clear history ─────────────────────────────────────────────────────────────
def clear_history():
    get_agent().clear_history()
    return [], "✅ Conversation history cleared (long-term memory kept)."


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="🆓 Free Claude Agent") as demo:

    gr.Markdown("""
# 🆓 Free Claude Agent
*Multi-provider • Token-optimized • Real conversation memory • 100% free*

> **Providers active:** Groq · Sambanova · Cerebras (whichever keys are set)  
> **Smart routing:** Simple queries → fast 8B model · Complex queries → 70B model
    """)

    # Main chat
    chatbot = gr.ChatInterface(
        fn=chat_stream,
        title="",
        examples=[
            "How do I fix a React re-render bug?",
            "Explain quantum entanglement simply",
            "Debug this Python error: IndexError: list index out of range",
            "Write a Python function to parse JSON safely with fallback",
            "Compare PostgreSQL vs MongoDB for a social app",
        ],
        cache_examples=False,
    )

    # Controls row
    with gr.Row():
    clear_btn = gr.Button("🗑️ Clear History", variant="secondary", scale=1)
    clear_status = gr.Markdown()

    clear_btn.click(
        fn=clear_history,
        outputs=[chatbot.chatbot, clear_status]
    )

    # Usage accordion
    with gr.Accordion("📊 Usage Stats", open=False):
        usage_btn = gr.Button("🔄 Refresh Stats")
        usage_display = gr.Markdown()
        usage_btn.click(fn=show_usage, outputs=usage_display)

    gr.Markdown("""
---
### 💡 How it works
| Feature | Detail |
|---|---|
| **Caveman mode** | Strips filler words → ~75% token savings |
| **Smart routing** | ⚡ 8B for simple · 🧠 70B for reasoning/code |
| **Multi-turn memory** | Last 12 turns kept live in context |
| **Long-term memory** | Older turns compressed → SQLite, recalled by BM25 |
| **Multi-provider fallback** | Groq → Sambanova → Cerebras (auto) |

### 🔧 Self-host
```bash
git clone https://github.com/danielwrites27-blip/free-claude-agent
cd free-claude-agent
cp .env.example .env   # add your API keys
docker build -t agent .
docker run -p 7860:7860 --env-file .env agent
```
    """)


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    auth = None
    hf_user = os.getenv("HF_AUTH_USERNAME")
    hf_pass = os.getenv("HF_AUTH_PASSWORD")
    if hf_user and hf_pass:          # FIXED: only set auth if BOTH are present
        auth = (hf_user, hf_pass)

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        auth=auth,
        theme=gr.themes.Soft(),
    )
