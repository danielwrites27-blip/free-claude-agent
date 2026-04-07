"""
Gradio interface for run.claw.cloud deployment.
Upgraded: Reasoning mode toggle, metrics display, working memory reset.
"""
import os
import gradio as gr
from src.agent import FreeAgent

agent = None

def get_agent():
    """Lazy-load agent with environment variables"""
    global agent
    if agent is None:
        agent = FreeAgent(
            api_key=os.getenv("GROQ_API_KEY"),
            daily_token_limit=int(os.getenv("DAILY_TOKEN_LIMIT", "50000")),
            memory_path=os.getenv("MEMORY_PATH", "agent_memory.db"),
            enable_reasoning=os.getenv("ENABLE_REASONING", "true").lower() == "true",
            enable_code_execution=False
        )
    return agent

def chat_with_agent(message: str, history: list,
                    use_reasoning: bool = False) -> str:
    """Gradio chat handler"""
    try:
        response = get_agent().ask(
            message,
            use_reasoning=use_reasoning
        )
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)[:200]}"

def show_usage() -> str:
    usage = get_agent().get_usage()
    return f"""
📊 **Token Usage**
- Used: {usage['tokens_used_today']:,} / {usage['daily_limit']:,}
- Remaining: {usage['remaining']:,}
- Resets in: ~{usage['reset_in_hours']}h
"""

def show_metrics() -> str:
    metrics = get_agent().get_metrics()
    if not metrics:
        return "📈 No queries yet"
    return f"""
📈 **Performance Metrics**
- Total queries: {metrics.get('total_queries', 0)}
- Success rate: {metrics.get('success_rate', 0)*100:.1f}%
- Avg tokens/query: {metrics.get('avg_tokens_per_query', 0):.0f}
- Avg latency: {metrics.get('avg_latency_ms', 0):.0f}ms
"""

def reset_memory() -> str:
    get_agent().reset_working_memory()
    return "🧹 Working memory cleared."

# Gradio interface
with gr.Blocks(title="🆓 Free Claude Agent") as demo:
    gr.Markdown("""
    ## 🆓 Free Claude-Style Agent
    *100% free • Groq + Cerebras + SambaNova • Token-optimized*
    """)

    reasoning_toggle = gr.Checkbox(
        label="🧠 Deep Reasoning",
        value=False,
        info="Chain-of-thought for complex problems"
    )

    chatbot = gr.ChatInterface(
        fn=lambda msg, hist: chat_with_agent(msg, hist, reasoning_toggle.value),
        title="Ask anything",
        examples=[
            "Hello, who are you?",
            "What is 15% of 240?",
            "Explain quantum entanglement simply",
            "Debug this Python error: IndexError: list index out of range",
            "If all A are B and all B are C, are all A definitely C?"
        ]
    )

    with gr.Accordion("📊 Usage & Metrics", open=False):
        with gr.Row():
            usage_btn = gr.Button("🔄 Refresh Usage")
            metrics_btn = gr.Button("📈 Show Metrics")
            reset_btn = gr.Button("🧹 Reset Memory")

        usage_display = gr.Markdown()
        metrics_display = gr.Markdown()
        reset_display = gr.Markdown()

        usage_btn.click(show_usage, outputs=usage_display)
        metrics_btn.click(show_metrics, outputs=metrics_display)
        reset_btn.click(reset_memory, outputs=reset_display)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        auth=(os.getenv("HF_AUTH_USERNAME"), os.getenv("HF_AUTH_PASSWORD"))
             if os.getenv("HF_AUTH_USERNAME") and os.getenv("HF_AUTH_PASSWORD")
             else None
    )
