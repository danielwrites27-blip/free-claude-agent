"""
seed_memory.py — Pre-seed local ChromaDB with facts tested by memory.json evals.

Run this BEFORE running memory evals locally:
    python3 tests/seed_memory.py

Then run evals:
    python3 tests/run_evals.py --no-judge --delay 20 --category memory

NOTE: mem_001 and mem_005 are live store-in-session evals — no seeding needed.
      All other memory evals require these facts to already exist in ChromaDB.
"""

import os
import sys

# Ensure we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# IMPORTANT: Set CHROMA_PATH env var BEFORE importing TokenEfficientMemory.
# _init_db() reads os.getenv("CHROMA_PATH") at init time — must be set first.
CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chromadb"))
os.makedirs(CHROMA_PATH, exist_ok=True)
os.environ["CHROMA_PATH"] = CHROMA_PATH
print(f"[seed_memory] ChromaDB path: {CHROMA_PATH}")

from src.memory import TokenEfficientMemory

mem = TokenEfficientMemory()

facts = [
    # mem_002 — project name + platform
    "User's project is called free-claude-agent and it runs on Kubernetes.",

    # mem_003 — user identity
    "User's name is Daniel.",

    # mem_004 — UI preference
    "User prefers dark mode for the UI theme.",

    # mem_006 — ChromaDB keyword match
    "This project uses ChromaDB for persistent vector memory. ChromaDB is stored at /app/data/chromadb on the pod and ./data/chromadb locally.",

    # mem_007 — semantic: how project stores info between conversations
    "The project stores information between conversations using ChromaDB, a persistent vector database. Older conversation turns are summarized and stored in ChromaDB so the agent can recall them in future sessions.",

    # mem_008 — GROQ_API_KEY exact term
    "GROQ_API_KEY is the primary fallback API key for the project. It uses Groq's llama-3.1-8b-instant model as the ultimate fallback provider. Note: Groq API is Cloudflare-blocked on the VM IP — use the pod for anything needing Groq.",
]

print(f"[seed_memory] Seeding {len(facts)} facts...\n")

for i, fact in enumerate(facts, 1):
    try:
        mem.store(fact)
        print(f"  ✅ [{i}/{len(facts)}] Stored: {fact[:80]}{'...' if len(fact) > 80 else ''}")
    except Exception as e:
        print(f"  ❌ [{i}/{len(facts)}] FAILED: {e}")
        print(f"             Fact: {fact[:80]}")

print("\n[seed_memory] Done. Run evals now:")
print("  python3 tests/run_evals.py --no-judge --delay 20 --category memory")