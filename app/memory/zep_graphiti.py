from typing import Optional, List
import httpx
from app.config import config_yaml
from app.memory.sqlite_graph_store import SQLiteGraphStore
from app.utils.logger import get_logger

logger = get_logger("zep_graphiti")

class ZepGraphitiMemory:
    """Tier-2 Zep Graphiti Temporal Knowledge Graph integration.
    
    Supports dual-mode execution:
    1. Online Zep Graph API (http://localhost:8080)
    2. Durable SQLite WAL Graph Store (Zero-dependency local persistence)
    """

    def __init__(self, zep_url: Optional[str] = None):
        self.zep_url = zep_url or config_yaml.get("memory", {}).get("zep", {}).get("url", "http://localhost:8080")
        self.sqlite_store = SQLiteGraphStore()

    async def get_session_context(self, session_id: str) -> Optional[str]:
        """Fetch temporal entity relationship facts for a session."""
        if not session_id:
            return None

        # 1. Try Live Zep API Server
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{self.zep_url}/api/v1/sessions/{session_id}/facts")
                if res.status_code == 200:
                    data = res.json()
                    facts = data.get("facts", [])
                    if facts:
                        return "\n".join(f"- {f}" for f in facts)
        except Exception:
            pass

        # 2. Fallback: Query Durable SQLite Graph Store
        session_facts = await self.sqlite_store.get_facts(session_id, limit=5)
        if session_facts:
            return "\n".join(f"- {f}" for f in session_facts)

        return None

    async def add_session_interaction(self, session_id: str, user_prompt: str, response: str) -> bool:
        """Asynchronously log new interaction to Zep graph store & durable SQLite graph store."""
        if not session_id:
            return False

        # Extract temporal fact from prompt/response pair
        fact = f"User asked: '{user_prompt}' -> Gateway response context: '{response[:150]}...'"
        
        # Persist fact to durable SQLite store
        await self.sqlite_store.add_fact(session_id, fact)

        # Log to live Zep Graph API if available
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.post(
                    f"{self.zep_url}/api/v1/sessions/{session_id}/messages",
                    json={
                        "messages": [
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": response}
                        ]
                    }
                )
                return res.status_code in (200, 201)
        except Exception:
            pass

        return True

    async def clear(self) -> bool:
        """Clear all session fact knowledge graphs from durable storage."""
        await self.sqlite_store.clear()
        return True
