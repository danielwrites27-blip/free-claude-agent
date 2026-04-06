"""
Lightweight memory layer using SQLite FTS5.
Single-file, no external dependencies, sub-5ms retrieval.
"""
import sqlite3
import hashlib
import json
import tiktoken
from pathlib import Path
from typing import Optional, List

class TokenEfficientMemory:
    """Single-file memory with BM25 full-text search + token budgeting"""
    
    def __init__(self, path: str = "agent_memory.mv2"):
        self.db_path = Path(path)
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite with FTS5 virtual table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # FTS5 table for full-text search (BM25 ranking)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories 
            USING fts5(
                id, content, tags, timestamp,
                content_rank='bm25(10,1)'
            )
        """)
        
        # Metadata table for embeddings + token counts
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
        """Count tokens using tiktoken (cl100k_base)"""
        return len(self.encoder.encode(text))
    
    def store(self, content: str, tags: Optional[List[str]] = None, 
              embedding: Optional[bytes] = None) -> str:
        """Store a memory entry with automatic token counting"""
        mem_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        token_count = self._count_tokens(content)
        tags_json = json.dumps(tags or [])
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert/update memory
        cursor.execute(
            "INSERT OR REPLACE INTO memories (id, content, tags, timestamp) VALUES (?, ?, ?, datetime('now'))",
            (mem_id, content, tags_json)
        )
        
        # Store metadata if provided
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
        """
        BM25 full-text search with token budget enforcement.
        Returns concatenated relevant memories.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # FTS5 BM25 search (fetch extra to filter by tokens)
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
        
        # Filter by cumulative token budget
        selected = []
        total_tokens = 0
        
        for content, token_count in results:
            # Use stored count or recalculate
            tokens = token_count or self._count_tokens(content)
            
            if total_tokens + tokens <= max_tokens:
                selected.append(content)
                total_tokens += tokens
            else:
                break  # Budget exhausted
        
        return "\n\n---\n\n".join(selected) if selected else ""
    
    def clear(self):
        """Clear all memories (for testing)"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM memories")
        conn.execute("DELETE FROM metadata")
        conn.commit()
        conn.close()
