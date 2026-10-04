from typing import Optional, Dict, Any
import httpx
from app.config import settings, config_yaml

class ZepGraphitiMemory:
    """Tier-2 Zep Graphiti Temporal Knowledge Graph integration."""

    def __init__(self, zep_url: Optional[str] = None):
        self.zep_url = zep_url or config_yaml.get("memory", {}).get("zep", {}).get("url", "http://localhost:8080")

    async def get_session_context(self, session_id: str) -> Optional[str]:
        """Fetch temporal entity relationship facts for a session."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.zep_url}/api/v1/sessions/{session_id}/facts")
                if res.status_code == 200:
                    data = res.json()
                    facts = data.get("facts", [])
                    return "\n".join(facts)
        except Exception:
            pass
        return None

    async def add_session_interaction(self, session_id: str, user_prompt: str, response: str) -> bool:
        """Asynchronously log new interaction to Zep graph store."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
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
            return False
