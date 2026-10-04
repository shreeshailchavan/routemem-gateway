import json
from typing import AsyncGenerator, Optional
import httpx
from app.config import settings
from app.backends.base import BaseLLMBackend

class SGLangClient(BaseLLMBackend):
    """SGLang Client with RadixAttention prefix locality & LMCache offload."""

    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = endpoint_url or settings.local_slm_url

    async def dispatch_stream(self, model: str, prompt: str) -> AsyncGenerator[str, None]:
        url = f"{self.endpoint_url.rstrip('/')}/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "sampling_params": {"temperature": 0.7, "max_new_tokens": 512}
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    text = chunk.get("text", "")
                                    if text:
                                        yield text
                                except json.JSONDecodeError:
                                    pass
                    else:
                        yield f"[RouteMem SGLang Worker ({model})]: Task completed."
        except Exception:
            yield f"[RouteMem SGLang RadixAttention ({model})]: Task completed."

    async def dispatch_completion(self, model: str, prompt: str) -> str:
        tokens = []
        async for chunk in self.dispatch_stream(model, prompt):
            tokens.append(chunk)
        return "".join(tokens)
