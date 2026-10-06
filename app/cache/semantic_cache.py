import uuid
import httpx
from typing import Optional, List
from pydantic import BaseModel

from app.config import settings
from app.cache.dense_embedder import DenseEmbedder
from app.utils.logger import get_logger

logger = get_logger("semantic_cache")

class SemanticHit(BaseModel):
    response: str
    similarity_score: float
    prompt_id: str

class SemanticCache:
    """Tier-1 Semantic Vector Cache using Qdrant REST API cosine similarity search."""

    def __init__(self, qdrant_url: Optional[str] = None, collection_name: str = "routemem_semantic_cache"):
        self.qdrant_url = (qdrant_url or settings.qdrant_url).rstrip('/')
        self.collection_name = collection_name
        self.embedder = DenseEmbedder()
        self.vector_dim = self.embedder.vector_dim
        self._collection_ensured = False

    async def _ensure_collection(self, client: httpx.AsyncClient):
        if self._collection_ensured:
            return
        try:
            res = await client.get(f"{self.qdrant_url}/collections/{self.collection_name}")
            if res.status_code == 404:
                payload = {
                    "vectors": {
                        "size": self.vector_dim,
                        "distance": "Cosine"
                    }
                }
                await client.put(f"{self.qdrant_url}/collections/{self.collection_name}", json=payload)
            self._collection_ensured = True
        except Exception as e:
            logger.error(f"Qdrant ensure collection error: {e}")

    def _embed_text(self, text: str) -> List[float]:
        return self.embedder.embed(text)

    async def search(self, user_prompt: str, threshold: Optional[float] = None) -> Optional[SemanticHit]:
        if not settings.semantic_cache_enabled:
            return None
        target_threshold = threshold or settings.semantic_cache_threshold
        try:
            vector = self._embed_text(user_prompt)
            async with httpx.AsyncClient(timeout=3.0) as client:
                await self._ensure_collection(client)
                search_payload = {
                    "vector": vector,
                    "limit": 1,
                    "score_threshold": target_threshold,
                    "with_payload": True
                }
                res = await client.post(
                    f"{self.qdrant_url}/collections/{self.collection_name}/points/search",
                    json=search_payload
                )
                if res.status_code == 200:
                    data = res.json()
                    results = data.get("result", [])
                    if results and len(results) > 0:
                        hit = results[0]
                        payload = hit.get("payload") or {}
                        return SemanticHit(
                            response=payload.get("response", ""),
                            similarity_score=hit.get("score", 0.0),
                            prompt_id=str(hit.get("id", ""))
                        )
            return None
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return None

    async def index(self, user_prompt: str, response: str) -> bool:
        if not settings.semantic_cache_enabled:
            return False
        try:
            vector = self._embed_text(user_prompt)
            point_id = str(uuid.uuid4())
            async with httpx.AsyncClient(timeout=5.0) as client:
                await self._ensure_collection(client)
                points_payload = {
                    "points": [
                        {
                            "id": point_id,
                            "vector": vector,
                            "payload": {
                                "user_prompt": user_prompt,
                                "response": response
                            }
                        }
                    ]
                }
                res = await client.put(
                    f"{self.qdrant_url}/collections/{self.collection_name}/points",
                    json=points_payload
                )
                return res.status_code in (200, 201, 202)
        except Exception as e:
            logger.error(f"Qdrant indexing error: {e}")
            return False

    async def clear(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.delete(f"{self.qdrant_url}/collections/{self.collection_name}")
                payload = {
                    "vectors": {
                        "size": self.vector_dim,
                        "distance": "Cosine"
                    }
                }
                await client.put(f"{self.qdrant_url}/collections/{self.collection_name}", json=payload)
                self._collection_ensured = True
                return True
        except Exception as e:
            logger.error(f"Qdrant clear error: {e}")
            return False

    async def close(self):
        pass
