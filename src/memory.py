"""
Lightweight memory layer using SQLite FTS5.
Single-file, no external dependencies, sub-5ms retrieval.
Plus: Working memory for multi-turn reasoning.
"""
import sqlite3
import hashlib
import json
import tiktoken
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime


class TokenEfficientMemory:
    """Single-file memory with BM25 full-text search + token budgeting"""

    def __init__(self, path: str = "agent_memory.db"):
        self.db_path = Path(path)
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self._init_db()

    def _init_db(self):
        """Initialize SQLite with FTS5 virtual table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories
            USING fts5(
                id, content, tags, timestamp
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id TEXT PRIMARY KEY,
                embedding BLOB,
                token_count INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def store(self, content: str, tags: Optional[List[str]] = None,
              embedding: Optional[bytes] = None) -> str:
        """Store a memory entry with automatic token counting"""
        mem_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        token_count = self._count_tokens(content)
        tags_json = json.dumps(tags or [])

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT OR REPLACE INTO memories (id, content, tags, timestamp) VALUES (?, ?, ?, datetime('now'))",
            (mem_id, content, tags_json)
        )

        if embedding:
            cursor.execute(
                "INSERT OR REPLACE INTO metadata (id, embedding, token_count) VALUES (?, ?, ?)",
                (mem_id, embedding, token_count)
            )

        conn.commit()
        conn.close()
        return mem_id

    def recall(self, query: str, top_k: int = 3,
               max_tokens: int = 2000) -> str:
        """BM25 full-text search with token budget enforcement"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT content, token_count
            FROM memories
            LEFT JOIN metadata ON memories.id = metadata.id
            WHERE memories MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, top_k * 2))

        results = cursor.fetchall()
        conn.close()

        selected = []
        total_tokens = 0

        for content, token_count in results:
            tokens = token_count or self._count_tokens(content)
            if total_tokens + tokens <= max_tokens:
                selected.append(content)
                total_tokens += tokens
            else:
                break

        return "\n\n---\n\n".join(selected) if selected else ""

    def clear(self):
        """Clear all memories"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM memories")
        conn.execute("DELETE FROM metadata")
        conn.commit()
        conn.close()


class WorkingMemory:
    """Short-term reasoning state for current conversation"""

    def __init__(self, max_items: int = 10):
        self.items: List[Dict] = []
        self.max_items = max_items

    def add(self, step: str, result: str, confidence: float = 0.8,
            metadata: Optional[Dict] = None):
        item = {
            "step": step,
            "result": result[:500],
            "confidence": confidence,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        self.items.append(item)
        if len(self.items) > self.max_items:
            self.items.pop(0)

    def get_context(self) -> str:
        if not self.items:
            return ""
        lines = ["<working_memory>"]
        for item in self.items:
            conf_icon = "✓" if item["confidence"] > 0.7 else "⚠"
            lines.append(f"{conf_icon} {item['step']}: {item['result']}")
        lines.append("</working_memory>")
        return "\n".join(lines)

    def clear(self):
        self.items = []

    def to_dict(self) -> List[Dict]:
        return self.items.copy()

    @classmethod
    def from_dict(cls, data: List[Dict]) -> 'WorkingMemory':
        wm = cls()
        wm.items = data
        return wm


class ConversationSummarizer:
    """Compress long conversations into token-efficient summaries"""

    def __init__(self, max_summary_tokens: int = 500):
        self.max_tokens = max_summary_tokens

    def summarize(self, messages: List[Dict], generate_fn) -> str:
        if not messages:
            return ""

        formatted = []
        for msg in messages[-20:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content'][:300]}")

        summary_prompt = f"""Summarize this conversation in 3-5 bullet points.
Focus on: decisions made, code written, errors fixed, open questions.

Conversation:
{chr(10).join(formatted)}

Summary (concise, factual):
"""
        try:
            return generate_fn(summary_prompt, max_tokens=self.max_tokens).strip()
        except Exception:
            return "Conversation summary unavailable."
