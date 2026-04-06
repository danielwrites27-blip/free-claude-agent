"""
Gradio interface for Hugging Face Spaces deployment.
Run locally: python app.py
Deploy: GitHub Actions auto-pushes to HF Spaces
"""
import os
import gradio as gr

from src.agent import FreeAgent

# Initialize agent (lazy load on first request)
agent = None

def get_agent():
    """Lazy-load agent with environment variables"""
    global agent
    if agent is None:
        agent = FreeAgent(
            api_key=os.getenv("GROQ_API_KEY"),
            daily_token_limit=int(os.getenv("DAILY_TOKEN_LIMIT", "50000")),
            memory_path=os.getenv("MEMORY_PATH", "agent_memory.mv2")
        )
    return agent

def chat_with_agent(message: str, history: list) -> str:
    """Gradio chat handler"""
    try:
        response = get_agent().ask(message)
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)[:200]}"

def show_usage() -> str:
    """Display current token usage"""
    usage = get_agent().get_usage()
    return f"""
    📊 **Token Usage**
    - Used: {usage['tokens_used_today']:,} / {usage['daily_limit']:,}
    - Remaining: {usage['remaining']:,}
    - Resets in: ~{usage['reset_in_hours']}h
    """

# Gradio interface
with gr.Blocks(title="🆓 Free Claude Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    ## 🆓 Free Claude-Style Agent
    *100% free • Groq API • Token-optimized • No credit card*
    
    🔑 Get free API key: [console.groq.com/keys](https://console.groq.com/keys)
    """)
    
    chatbot = gr.ChatInterface(
        fn=chat_with_agent,
        title="Ask anything",
        examples=[
            "How do I fix a React re-render bug?",
            "Explain quantum entanglement simply",
            "Debug this Python error: IndexError: list index out of range"
        ],
        theme=gr.themes.Soft()
    )
    
    with gr.Accordion("📊 Usage Stats", open=False):
        usage_btn = gr.Button("Refresh")
        usage_display = gr.Markdown()
        usage_btn.click(show_usage, outputs=usage_display)
    
    gr.Markdown("""
    ### 💡 Tips
    - **Caveman mode**: Outputs are compressed (~75% token savings)
    - **Smart routing**: Simple queries use 8B model, complex use 70B
    - **Memory**: Conversations auto-save to local SQLite
    - **Free tier**: ~1,000 requests/day with 70B model
    
    ### 🔧 Self-host
    ```bash
    git clone https://github.com/YOU/free-claude-agent
    cd free-claude-agent
    cp .env.example .env  # Add your GROQ_API_KEY
    docker build -t agent .
    docker run -p 7860:7860 --env-file .env agent
    ```
    """)

if __name__ == "__main__":
    # For Hugging Face Spaces: use server_port=7860, server_name="0.0.0.0"
    demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    auth=(os.getenv("HF_AUTH_USERNAME"), os.getenv("HF_AUTH_PASSWORD"))
         if os.getenv("HF_AUTH_USERNAME") and os.getenv("HF_AUTH_PASSWORD") else None
)
