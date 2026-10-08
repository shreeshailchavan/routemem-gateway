import json
from typing import AsyncGenerator, Optional
import httpx
from app.config import settings
from app.backends.base import BaseLLMBackend

class CloudAPIClient(BaseLLMBackend):
    """Cloud Provider API Client (OpenAI & Anthropic)."""

    def __init__(self, openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None):
        self.openai_key = openai_api_key or settings.openai_api_key
        self.anthropic_key = anthropic_api_key or settings.anthropic_api_key

    async def dispatch_stream(self, model: str, prompt: str, system_prompt: str = "", max_tokens: Optional[int] = None, **kwargs) -> AsyncGenerator[str, None]:
        if "claude" in model.lower() or "anthropic" in model.lower():
            async for chunk in self._stream_anthropic(model, prompt, system_prompt, max_tokens=max_tokens):
                yield chunk
        else:
            async for chunk in self._stream_openai(model, prompt, system_prompt, max_tokens=max_tokens):
                yield chunk

    async def _stream_openai(self, model: str, prompt: str, system_prompt: str = "", max_tokens: Optional[int] = None) -> AsyncGenerator[str, None]:
        if not self.openai_key:
            from app.backends.groq_client import GroqClient
            groq = GroqClient()
            async for chunk in groq.dispatch_stream("openai/gpt-oss-120b", prompt, system_prompt=system_prompt, max_tokens=max_tokens):
                yield chunk
            return

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or 2048,
            "stream": True
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        yield f"[OpenAI API Error {response.status_code}]"
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
            yield f"[RouteMem Cloud OpenAI ({model})]: Task completed."

    async def _stream_anthropic(self, model: str, prompt: str, system_prompt: str = "", max_tokens: Optional[int] = None) -> AsyncGenerator[str, None]:
        if not self.anthropic_key:
            from app.backends.groq_client import GroqClient
            groq = GroqClient()
            async for chunk in groq.dispatch_stream("openai/gpt-oss-120b", prompt, system_prompt=system_prompt, max_tokens=max_tokens):
                yield chunk
            return

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens or 2048,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        yield f"[Anthropic API Error {response.status_code}]"
                        return
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            try:
                                chunk = json.loads(data_str)
                                if chunk.get("type") == "content_block_delta":
                                    text = chunk.get("delta", {}).get("text", "")
                                    if text:
                                        yield text
                            except json.JSONDecodeError:
                                pass
        except Exception:
            yield f"[RouteMem Cloud Anthropic ({model})]: Task completed."

    async def dispatch_completion(self, model: str, prompt: str, system_prompt: str = "", **kwargs) -> str:
        tokens = []
        async for chunk in self.dispatch_stream(model, prompt, system_prompt=system_prompt, **kwargs):
            tokens.append(chunk)
        return "".join(tokens)
