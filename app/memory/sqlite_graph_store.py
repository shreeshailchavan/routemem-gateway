import os
import sqlite3
import time
from pathlib import Path
from typing import List, Optional
from app.utils.logger import get_logger

logger = get_logger("sqlite_graph_store")

DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_DB_PATH = DB_DIR / "graphiti_memory.db"

class SQLiteGraphStore:
    """Thread-safe, high-concurrency SQLite store for Zep Graphiti session facts with WAL mode."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(self.db_path.parent, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS session_facts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        fact TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON session_facts (session_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON session_facts (created_at);")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite Graph Store at {self.db_path}: {e}")

    async def add_fact(self, session_id: str, fact: str) -> bool:
        """Persist a new session fact to SQLite."""
        if not session_id or not fact:
            return False
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO session_facts (session_id, fact, created_at) VALUES (?, ?, ?)",
                    (session_id, fact, time.time())
                )
            return True
        except Exception as e:
            logger.error(f"Error persisting fact to SQLite for session {session_id}: {e}")
            return False

    async def get_facts(self, session_id: str, limit: int = 5) -> List[str]:
        """Retrieve the most recent facts for a session."""
        if not session_id:
            return []
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT fact FROM session_facts WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, limit)
                )
                rows = cursor.fetchall()
                # Return in chronological order
                return [r[0] for r in reversed(rows)]
        except Exception as e:
            logger.error(f"Error reading facts from SQLite for session {session_id}: {e}")
            return []

    async def clear(self, session_id: Optional[str] = None) -> bool:
        """Clear session facts (either for a single session or all sessions)."""
        try:
            with self._get_connection() as conn:
                if session_id:
                    conn.execute("DELETE FROM session_facts WHERE session_id = ?", (session_id,))
                else:
                    conn.execute("DELETE FROM session_facts;")
            return True
        except Exception as e:
            logger.error(f"Error clearing facts from SQLite: {e}")
            return False

    async def get_total_facts_count(self) -> int:
        """Return total number of facts stored across all sessions."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM session_facts;")
                row = cursor.fetchone()
                return row[0] if row else 0
        except Exception as e:
            logger.error(f"Error getting total facts count: {e}")
            return 0
