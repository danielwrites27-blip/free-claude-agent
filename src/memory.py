"""
Semantic memory layer using ChromaDB + sentence-transformers.
Replaces SQLite FTS5 keyword search with vector similarity search.
Same interface as TokenEfficientMemory — drop-in replacement.
"""
import hashlib
import json
import tiktoken
from typing import Optional, List, Dict
from datetime import datetime


class TokenEfficientMemory:
    """Semantic memory with ChromaDB vector search + token budgeting.
    Drop-in replacement for SQLite FTS5 version — same store()/recall() interface.
    """

    def __init__(self, path: str = "agent_memory.db"):
        self.path = path
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.collection = None
        self._init_db()

    def _init_db(self):
        """Initialize ChromaDB with local sentence-transformer embeddings"""
        try:
            import chromadb
            # EphemeralClient = in-memory, no persistence needed
            # (pod restarts wipe disk anyway — same behavior as before)
            self.client = chromadb.EphemeralClient()
            self.collection = self.client.get_or_create_collection(
                name="agent-memories",
                metadata={"hnsw:space": "cosine"}
                # Default embedding: all-MiniLM-L6-v2 (local, free, no API key)
            )
            print("[Memory] ChromaDB semantic memory initialized", flush=True)
        except Exception as e:
            print(f"[Memory] ChromaDB init failed: {e} — falling back to keyword search", flush=True)
            self.collection = None
            self._init_fallback()

    def _init_fallback(self):
        """SQLite FTS5 fallback if ChromaDB fails"""
        import sqlite3
        from pathlib import Path
        self.db_path = Path(self.path)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories
            USING fts5(mem_id UNINDEXED, content, tags, timestamp UNINDEXED)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                mem_id TEXT PRIMARY KEY,
                token_count INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def store(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        token_count: Optional[int] = None,
    ) -> str:
        """Store a memory entry"""
        mem_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        actual_token_count = token_count or self._count_tokens(content)
        tags_str = json.dumps(tags or [])

        if self.collection is not None:
            # ChromaDB path — upsert handles duplicates automatically
            try:
                self.collection.upsert(
                    ids=[mem_id],
                    documents=[content],
                    metadatas=[{
                        "tags": tags_str,
                        "token_count": actual_token_count,
                        "timestamp": datetime.now().isoformat()
                    }]
                )
            except Exception as e:
                print(f"[Memory] ChromaDB store failed: {e}", flush=True)
        else:
            # SQLite fallback path
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories WHERE mem_id = ?", (mem_id,))
            cursor.execute(
                "INSERT INTO memories (mem_id, content, tags, timestamp) VALUES (?, ?, ?, datetime('now'))",
                (mem_id, content, tags_str)
            )
            cursor.execute(
                "INSERT OR REPLACE INTO metadata (mem_id, token_count) VALUES (?, ?)",
                (mem_id, actual_token_count)
            )
            conn.commit()
            conn.close()

        return mem_id

    def recall(self, query: str, top_k: int = 3, max_tokens: int = 2000) -> str:
        """Semantic similarity search with token budget enforcement"""
        if self.collection is not None:
            # ChromaDB semantic search path
            try:
                count = self.collection.count()
                if count == 0:
                    return ""
                # Query returns most semantically similar memories
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k * 2, count),
                    include=["documents", "metadatas", "distances"]
                )
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]

                selected = []
                total_tokens = 0
                for doc, meta in zip(docs, metas):
                    tokens = meta.get("token_count", self._count_tokens(doc))
                    if total_tokens + tokens <= max_tokens:
                        selected.append(doc)
                        total_tokens += tokens
                    else:
                        break

                return "\n\n---\n\n".join(selected) if selected else ""

            except Exception as e:
                print(f"[Memory] ChromaDB recall failed: {e}", flush=True)
                return ""
        else:
            # SQLite fallback path
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT m.content, md.token_count
                    FROM memories m
                    LEFT JOIN metadata md ON m.mem_id = md.mem_id
                    WHERE memories MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, top_k * 2))
                results = cursor.fetchall()
            except sqlite3.OperationalError:
                results = []
            finally:
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
        if self.collection is not None:
            try:
                self.client.delete_collection("agent-memories")
                self.collection = self.client.get_or_create_collection(
                    name="agent-memories",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                print(f"[Memory] ChromaDB clear failed: {e}", flush=True)
        else:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM memories")
            conn.execute("DELETE FROM metadata")
            conn.commit()
            conn.close()

    def get_recent(self, n: int = 5) -> List[str]:
        """Get n most recent memory entries"""
        if self.collection is not None:
            try:
                results = self.collection.get(
                    include=["documents", "metadatas"]
                )
                docs = results.get("documents", [])
                metas = results.get("metadatas", [])
                # Sort by timestamp descending
                paired = sorted(
                    zip(docs, metas),
                    key=lambda x: x[1].get("timestamp", ""),
                    reverse=True
                )
                return [doc for doc, _ in paired[:n]]
            except Exception:
                return []
        else:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM memories ORDER BY timestamp DESC LIMIT ?", (n,))
            rows = cursor.fetchall()
            conn.close()
            return [r[0] for r in rows]


# WorkingMemory and ConversationSummarizer unchanged — keep as-is
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
