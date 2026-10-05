import json
from typing import AsyncGenerator, Optional
import httpx
from app.config import settings
from app.backends.base import BaseLLMBackend

class VLLMClient(BaseLLMBackend):
    """vLLM OpenAI-compatible Client with prefix caching support."""

    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = endpoint_url or settings.local_slm_url

    async def dispatch_stream(self, model: str, prompt: str, system_prompt: str = "") -> AsyncGenerator[str, None]:
        url = f"{self.endpoint_url.rstrip('/')}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        yield f"vLLM Worker Status {response.status_code}"
                        return
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    yield delta
                            except json.JSONDecodeError:
                                pass
        except Exception:
            # When local vLLM GPU worker is offline, route to ultra-fast Groq LPU SLM (e.g. openai/gpt-oss-20b)
            from app.backends.groq_client import GroqClient
            groq = GroqClient()
            async for chunk in groq.dispatch_stream("openai/gpt-oss-20b", prompt, system_prompt=system_prompt):
                yield chunk

    async def dispatch_completion(self, model: str, prompt: str, system_prompt: str = "") -> str:
        tokens = []
        async for chunk in self.dispatch_stream(model, prompt, system_prompt=system_prompt):
            tokens.append(chunk)
        return "".join(tokens)
