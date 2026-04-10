import gradio as gr
import os
from src.agent import FreeAgent

# ─── Agent factory (singleton per session) ────────────────────────────────────
_agent_instance = None

import pathlib

def extract_file_text(filepath: str) -> str:
    """Extract plain text from .txt, .md, .pdf, or .docx files."""
    ext = pathlib.Path(filepath).suffix.lower()
    try:
        if ext in (".txt", ".md"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".pdf":
            import fitz  # pymupdf
            doc = fitz.open(filepath)
            return "\n\n".join(page.get_text() for page in doc)
        elif ext == ".docx":
            from docx import Document
            doc = Document(filepath)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            return f"[Unsupported file type: {ext}]"
    except Exception as e:
        return f"[Error reading file: {e}]"

def get_agent():
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = FreeAgent(
            api_key=os.getenv("GROQ_API_KEY"),
            daily_token_limit=int(os.getenv("DAILY_TOKEN_LIMIT", "50000")),
            memory_path=os.getenv("MEMORY_PATH", "agent_memory.db"),
        )
    return _agent_instance


def show_usage():
    agent = get_agent()
    used = agent.tokens_used_today
    limit = agent.daily_token_limit
    pct = min(int((used / limit) * 100), 100) if limit else 0
    color = "#4ade80" if pct < 70 else "#facc15" if pct < 90 else "#f87171"
    return f"""
    <div style="font-family:monospace;font-size:12px;color:#aaa;padding:4px 0;">
      <span style="color:#888;">Tokens today:</span>
      <span style="color:{color};font-weight:bold;">{used:,} / {limit:,}</span>
      <span style="color:#555;margin-left:8px;">({pct}%)</span>
      <div style="background:#333;border-radius:4px;height:6px;margin-top:4px;width:100%;">
        <div style="background:{color};border-radius:4px;height:6px;width:{pct}%;transition:width 0.3s;"></div>
      </div>
    </div>
    """


# ─── Main chat handler ─────────────────────────────────────────────────────────
def chat_stream(message, history, mode):
    agent = get_agent()
    caveman_mode = (mode == "Caveman")
    deep_reasoning = (mode == "Deep Reasoning")

    agent.caveman_mode = caveman_mode
    agent.deep_reasoning_mode = deep_reasoning

    full_response = ""
    for chunk in agent.ask_stream(message):
        full_response += chunk
        yield full_response


def get_status():
    agent = get_agent()
    model_label = getattr(agent, "_last_model_label", "⚡ ready")
    provider = getattr(agent, "_last_provider", "")
    label = f"`{model_label}`" + (f" · {provider}" if provider else "")
    return f"**Status:** {label}"


def clear_history():
    agent = get_agent()
    agent.conversation_history = []
    return [], show_usage()


# ─── Custom CSS ───────────────────────────────────────────────────────────────
css = """
/* Dark base */
body, .gradio-container {
    background: #0f1117 !important;
    color: #e2e8f0 !important;
}
.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
}

/* Chat bubbles */
.message.user {
    background: #1e2235 !important;
    border: 1px solid #2d3554 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}
.message.bot {
    background: #161b2e !important;
    border: 1px solid #1e2a45 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

/* Input box */
.gr-textbox textarea {
    background: #1a1f2e !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d3554 !important;
    border-radius: 8px !important;
}

/* Radio buttons */
.gr-radio label {
    color: #94a3b8 !important;
}
.gr-radio label:hover {
    color: #e2e8f0 !important;
}

/* Buttons */
button.primary {
    background: #3b5bdb !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
}
button.secondary {
    background: #1e2235 !important;
    border: 1px solid #2d3554 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
}
button.secondary:hover {
    border-color: #4a5568 !important;
    color: #e2e8f0 !important;
}

/* Header */
.header-row {
    border-bottom: 1px solid #1e2235;
    padding-bottom: 8px;
    margin-bottom: 4px;
}

/* Status bar */
.status-label {
    font-family: monospace;
    font-size: 12px;
    color: #64748b;
    padding: 2px 0;
}

/* Markdown */
.gr-markdown p, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: #e2e8f0 !important;
}
code {
    background: #1e2235 !important;
    color: #7dd3fc !important;
    padding: 2px 5px !important;
    border-radius: 4px !important;
}
"""


# ─── UI Layout ────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=css,
    title="Free Claude Agent 🤖",
) as demo:

    # Header
    with gr.Row(elem_classes="header-row"):
        gr.Markdown("## 🤖 Free Claude Agent")
        gr.Markdown(
            "<span style='color:#4a5568;font-size:13px;'>Multi-provider · Zero cost · Self-aware</span>",
            elem_classes="status-label"
        )

    # Chat
    chatbot = gr.Chatbot(
        label="",
        height=500,
        show_label=False,
        bubble_full_width=False,
        render_markdown=True,
    )

    # File upload
    file_upload = gr.File(
        label="📎 Attach file (.txt, .md, .pdf, .docx)",
        file_types=[".txt", ".md", ".pdf", ".docx"],
        scale=1,
        height=80,
    )

    # Input row
    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Ask anything… (Enter to send, Shift+Enter for newline)",
            show_label=False,
            scale=8,
            container=False,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

    # Controls row
    with gr.Row():
        mode_radio = gr.Radio(
            choices=["Normal", "Caveman", "Deep Reasoning"],
            value="Normal",
            label="Mode",
            scale=3,
            interactive=True,
        )
        with gr.Column(scale=2):
            clear_btn = gr.Button("🗑 Clear History", variant="secondary", size="sm")
            status_md = gr.Markdown("**Status:** `⚡ ready`", elem_classes="status-label")

    # Token usage bar
    usage_html = gr.HTML(show_usage())

    # ─── Event wiring ─────────────────────────────────────────────────────────

    def respond(message, history, mode, uploaded_file):
        history = history or []
        # Inject file content if a file is attached
        if uploaded_file is not None:
            file_text = extract_file_text(uploaded_file.name)
            filename = pathlib.Path(uploaded_file.name).name
            augmented_prompt = (
                f"[Uploaded file: {filename}]\n\n"
                f"{file_text}\n\n"
                f"---\nUser question: {message}"
            )
        else:
            augmented_prompt = message
        history.append([message, ""])  # show original message in chat, not the augmented one
        full = ""
        for chunk in chat_stream(augmented_prompt, history[:-1], mode):
            full = chunk
            history[-1][1] = full
            yield "", history, get_status(), show_usage(), None  # None clears the file upload

    msg_box.submit(
        respond,
        inputs=[msg_box, chatbot, mode_radio, file_upload],
        outputs=[msg_box, chatbot, status_md, usage_html, file_upload],
    )
    send_btn.click(
        respond,
        inputs=[msg_box, chatbot, mode_radio, file_upload],
        outputs=[msg_box, chatbot, status_md, usage_html, file_upload],
    )
    clear_btn.click(
        clear_history,
        outputs=[chatbot, usage_html],
    )


# ─── Auth & launch ────────────────────────────────────────────────────────────
auth_user = os.getenv("HF_AUTH_USERNAME")
auth_pass = os.getenv("HF_AUTH_PASSWORD")

launch_kwargs = dict(server_name="0.0.0.0", server_port=7860, show_error=True)
if auth_user and auth_pass:
    launch_kwargs["auth"] = (auth_user, auth_pass)

demo.launch(**launch_kwargs)
