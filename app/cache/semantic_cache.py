import uuid
from typing import Optional, List
from pydantic import BaseModel

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http import models as rest_models
except ImportError:
    AsyncQdrantClient = None
    rest_models = None

from app.config import settings
from app.cache.dense_embedder import DenseEmbedder

class SemanticHit(BaseModel):
    response: str
    similarity_score: float
    prompt_id: str

class SemanticCache:
    """Tier-1 Semantic Vector Cache using Qdrant HNSW cosine similarity search."""

    def __init__(self, qdrant_url: Optional[str] = None, collection_name: str = "routemem_semantic_cache"):
        self.qdrant_url = qdrant_url or settings.qdrant_url
        self.collection_name = collection_name
        self.embedder = DenseEmbedder()
        self.vector_dim = self.embedder.vector_dim
        self._client = None

    async def get_client(self):
        if AsyncQdrantClient is None:
            return None
        if self._client is None:
            self._client = AsyncQdrantClient(url=self.qdrant_url)
            await self._ensure_collection()
        return self._client

    async def _ensure_collection(self):
        if self._client is None or rest_models is None:
            return
        try:
            collections = await self._client.get_collections()
            collection_names = [c.name for c in collections.collections]
            if self.collection_name not in collection_names:
                await self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=self.vector_dim,
                        distance=rest_models.Distance.COSINE
                    )
                )
        except Exception:
            pass

    def _embed_text(self, text: str) -> List[float]:
        return self.embedder.embed(text)

    async def search(self, user_prompt: str, threshold: Optional[float] = None) -> Optional[SemanticHit]:
        if not settings.semantic_cache_enabled or AsyncQdrantClient is None:
            return None
        target_threshold = threshold or settings.semantic_cache_threshold
        try:
            client = await self.get_client()
            if client is None: return None
            vector = self._embed_text(user_prompt)
            results = await client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=1,
                score_threshold=target_threshold
            )
            if results and len(results) > 0:
                hit = results[0]
                payload = hit.payload or {}
                return SemanticHit(
                    response=payload.get("response", ""),
                    similarity_score=hit.score,
                    prompt_id=str(hit.id)
                )
            return None
        except Exception:
            return None

    async def index(self, user_prompt: str, response: str) -> bool:
        if not settings.semantic_cache_enabled or AsyncQdrantClient is None or rest_models is None:
            return False
        try:
            client = await self.get_client()
            if client is None: return False
            vector = self._embed_text(user_prompt)
            point_id = str(uuid.uuid4())
            await client.upsert(
                collection_name=self.collection_name,
                points=[
                    rest_models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "user_prompt": user_prompt,
                            "response": response
                        }
                    )
                ]
            )
            return True
        except Exception:
            return False

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None
