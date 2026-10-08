from typing import Optional, List, Set
import httpx
from app.config import settings, config_yaml
from app.memory.sqlite_graph_store import SQLiteGraphStore
from app.utils.logger import get_logger

logger = get_logger("zep_graphiti")

class ZepGraphitiMemory:
    """Tier-2 Zep Graphiti Temporal Knowledge Graph integration.
    
    Supports hybrid execution:
    1. Online Zep Cloud Context Graph API (https://api.getzep.com / Zep SDK)
    2. Direct REST Fallback with API-Key authentication
    3. Durable SQLite WAL Graph Store (Zero-latency local persistence)
    """

    def __init__(self, api_key: Optional[str] = None, zep_url: Optional[str] = None):
        self.api_key = (
            api_key
            or settings.zep_api_key
            or config_yaml.get("memory", {}).get("zep", {}).get("api_key", "")
        )
        self.zep_url = (
            zep_url
            or settings.zep_api_url
            or config_yaml.get("memory", {}).get("zep", {}).get("url", "https://api.getzep.com")
        ).rstrip("/")

        self.sqlite_store = SQLiteGraphStore()
        self._known_users: Set[str] = set()

        # Initialize AsyncZep Client if SDK available and api_key present
        self.client = None
        if self.api_key:
            try:
                from zep_cloud.client import AsyncZep
                self.client = AsyncZep(api_key=self.api_key)
                logger.info("Initialized live Zep Cloud Graphiti client.")
            except ImportError:
                logger.info("zep-cloud package not installed; falling back to direct REST client.")
            except Exception as e:
                logger.warning(f"Could not initialize Zep Cloud client: {e}")

    async def get_session_context(self, session_id: str, query: Optional[str] = None) -> Optional[str]:
        """Fetch temporal entity relationship facts for a session.
        
        Attempts retrieval from Zep Cloud Knowledge Graph, then falls back
        to durable local SQLite WAL store.
        """
        if not session_id:
            return None

        retrieved_facts: List[str] = []

        # 1. Query Zep Cloud Context Graph using SDK
        if self.client:
            try:
                search_query = query if query and len(query.strip()) > 3 else "context memory facts"
                results = await self.client.graph.search(
                    user_id=session_id,
                    query=search_query,
                    scope="edges"
                )
                if results and getattr(results, "edges", None):
                    for edge in results.edges:
                        fact_text = getattr(edge, "fact", None)
                        if fact_text and fact_text not in retrieved_facts:
                            retrieved_facts.append(fact_text)
                            if len(retrieved_facts) >= 5:
                                break
            except Exception as e:
                logger.debug(f"Zep Cloud graph search fallback: {e}")

        # 2. Query Zep Cloud / REST if SDK was unavailable but api_key / URL is configured
        if not retrieved_facts and self.api_key:
            try:
                headers = {
                    "Authorization": f"Api-Key {self.api_key}",
                    "User-Agent": "routemem-zep/1.0",
                    "Accept": "application/json"
                }
                async with httpx.AsyncClient(timeout=2.0) as http_client:
                    res = await http_client.get(
                        f"{self.zep_url}/api/v2/users/{session_id}/context",
                        headers=headers
                    )
                    if res.status_code == 200:
                        data = res.json()
                        ctx = data.get("context")
                        if ctx:
                            retrieved_facts.append(str(ctx))
            except Exception:
                pass

        # 3. Fallback / Merge: Query Durable SQLite Graph Store
        if not retrieved_facts:
            session_facts = await self.sqlite_store.get_facts(session_id, limit=5)
            if session_facts:
                retrieved_facts.extend(session_facts)

        if retrieved_facts:
            return "\n".join(f"- {f}" for f in retrieved_facts[:5])

        return None

    async def add_session_interaction(self, session_id: str, user_prompt: str, response: str) -> bool:
        """Asynchronously log new interaction to Zep Cloud graph store & durable SQLite store."""
        if not session_id:
            return False

        # Extract temporal fact string
        fact = f"User asked: '{user_prompt}' -> Gateway response context: '{response[:150]}...'"

        # Always persist to local SQLite store immediately (0ms blocking guarantee)
        await self.sqlite_store.add_fact(session_id, fact)

        # Ingest into live Zep Cloud Graph
        if self.client:
            try:
                # Ensure user exists in Zep Cloud
                if session_id not in self._known_users:
                    try:
                        await self.client.user.get(user_id=session_id)
                    except Exception:
                        await self.client.user.add(
                            user_id=session_id,
                            first_name="User",
                            last_name=session_id
                        )
                    self._known_users.add(session_id)

                # Ingest interaction into user's Context Graph
                await self.client.graph.add(
                    user_id=session_id,
                    type="text",
                    data=f"User interaction in session {session_id}:\nUser: {user_prompt}\nAssistant: {response[:350]}"
                )
                return True
            except Exception as e:
                logger.debug(f"Zep Cloud interaction ingest warning: {e}")

        # REST Ingest Fallback
        if self.api_key:
            try:
                headers = {
                    "Authorization": f"Api-Key {self.api_key}",
                    "User-Agent": "routemem-zep/1.0",
                    "Content-Type": "application/json"
                }
                async with httpx.AsyncClient(timeout=2.0) as http_client:
                    await http_client.post(
                        f"{self.zep_url}/api/v2/graphs/users/{session_id}/data",
                        headers=headers,
                        json={
                            "type": "text",
                            "data": f"User: {user_prompt}\nAssistant: {response[:350]}"
                        }
                    )
            except Exception:
                pass

        return True

    async def clear(self) -> bool:
        """Clear all session fact knowledge graphs from durable storage."""
        await self.sqlite_store.clear()
        self._known_users.clear()
        return True
