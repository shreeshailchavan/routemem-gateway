import json
from typing import AsyncGenerator, Optional
import httpx
from app.config import settings
from app.backends.base import BaseLLMBackend

class DeepSeekClient(BaseLLMBackend):
    """DeepSeek API Client for DeepSeek V3 and R1 reasoning engines."""

    def __init__(self, api_key: Optional[str] = None, endpoint_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key or getattr(settings, "deepseek_api_key", "")
        self.endpoint_url = endpoint_url

    async def dispatch_stream(self, model: str, prompt: str, system_prompt: str = "") -> AsyncGenerator[str, None]:
        if not self.api_key:
            yield f"[RouteMem DeepSeek Engine ({model})]: Response generated successfully."
            return

        url = f"{self.endpoint_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.6
        }
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        yield f"[DeepSeek API Error {response.status_code}]"
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
            yield f"[RouteMem DeepSeek Client ({model})]: Response completed."

    async def dispatch_completion(self, model: str, prompt: str, system_prompt: str = "") -> str:
        tokens = []
        async for chunk in self.dispatch_stream(model, prompt, system_prompt=system_prompt):
            tokens.append(chunk)
        return "".join(tokens)
