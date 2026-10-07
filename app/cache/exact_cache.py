import hashlib
from typing import Optional

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

from app.config import settings

class ExactCache:
    """Tier-0 Exact Hash Cache utilizing SHA-256 prompt keys in Redis."""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.redis_url
        self._client = None

    async def get_client(self):
        if redis is None:
            return None
        if self._client is None:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _compute_hash(self, system_prompt: str, user_prompt: str, context_prefix: str = "") -> str:
        payload = f"{system_prompt.strip()}::{context_prefix.strip()}::{user_prompt.strip()}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    async def get(self, system_prompt: str, user_prompt: str, context_prefix: str = "") -> Optional[str]:
        if not settings.exact_cache_enabled or redis is None:
            return None
        try:
            client = await self.get_client()
            if client is None: return None
            key = f"exact_cache:{self._compute_hash(system_prompt, user_prompt, context_prefix=context_prefix)}"
            return await client.get(key)
        except Exception:
            return None

    async def set(
        self,
        system_prompt: str,
        user_prompt: str,
        response: str,
        ttl_seconds: Optional[int] = None,
        context_prefix: str = ""
    ) -> bool:
        if not settings.exact_cache_enabled or redis is None:
            return False
        try:
            client = await self.get_client()
            if client is None: return False
            key = f"exact_cache:{self._compute_hash(system_prompt, user_prompt, context_prefix=context_prefix)}"
            ttl = ttl_seconds or settings.exact_cache_ttl_seconds
            await client.set(key, response, ex=ttl)
            return True
        except Exception:
            return False

    async def clear(self) -> bool:
        if redis is None:
            return False
        try:
            client = await self.get_client()
            if client:
                await client.flushdb()
                return True
        except Exception:
            pass
        return False

    async def close(self):
        if self._client and redis is not None:
            await self._client.aclose()
            self._client = None
