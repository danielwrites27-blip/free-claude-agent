"""
Caveman mode: Strip filler words, keep technical precision.
~75% token savings on outputs.
"""
import re

CAVEMAN_SYSTEM_PROMPT = """
You are in caveman mode. Follow these rules strictly:

1. NO filler: Remove "I'd be happy to", "Sure!", "Let me think", "The reason is"
2. NO articles: Skip "a", "an", "the" unless critical for meaning
3. NO hedging: Replace "might be worth considering" → "Do this:"
4. KEEP technical terms: "polymorphism", "useMemo", "OAuth2" stay exact
5. KEEP code blocks: Write full, correct code (caveman not stupid)
6. KEEP error messages: Quote exact strings
7. STOP when done: No summaries, no "let me know if you need more"

Example:
❌ Normal: "The issue you're experiencing is likely because you're creating a new object reference on each render. I'd recommend using useMemo to memoize the object."
✅ Caveman: "Inline object prop = new ref = re-render. Wrap in `useMemo`."

Output format: Direct answer. Then code if needed. Then stop.
"""

def compress_response(text: str) -> str:
    """Apply caveman rules to agent output"""
    if not text:
        return text
    
    # Remove common filler patterns (case-insensitive)
    fillers = [
        r"I['']d be happy to .*?\.",
        r"I would be happy to .*?\.",
        r"Sure!?\s*",
        r"Absolutely!?\s*",
        r"Let me (think|see|check|help).*?\.",
        r"The reason (this|you|that).*?is (that|because)\s*",
        r"It['']s worth noting that\s*",
        r"Please let me know if.*",
        r"Feel free to.*",
        r"Hope this helps!?\s*",
        r"In conclusion,?\s*",
        r"To summarize,?\s*",
    ]
    
    for pattern in fillers:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Remove articles (simple heuristic - skip in code blocks)
    # Split by code blocks to avoid modifying code
    parts = re.split(r'(```[\s\S]*?```)', text)
    processed = []
    
    for part in parts:
        if part.startswith('```') and part.endswith('```'):
            # Keep code blocks intact
            processed.append(part)
        else:
            # Remove articles outside code
            part = re.sub(r'\b(a|an|the)\b', '', part, flags=re.IGNORECASE)
            processed.append(part)
    
    text = ''.join(processed)
    
    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r' {2,}', ' ', text)       # Single spaces
    text = text.strip()
    
    return text
