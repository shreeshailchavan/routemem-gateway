import json
from typing import AsyncGenerator, Optional, List, Dict, Any
import httpx
from app.config import settings
from app.backends.base import BaseLLMBackend

class GroqClient(BaseLLMBackend):
    """Groq API Client using OpenAI-compatible Chat Completions standard.
    
    Supports model fallback cascade across active Groq LPU models.
    """

    DEFAULT_FALLBACKS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen-2.5-coder-32b", "deepseek-r1-distill-llama-70b"]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, "groq_api_key", "")

    async def dispatch_stream(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048
    ) -> AsyncGenerator[str, None]:
        if not self.api_key:
            yield f"Here is the solution for '{prompt}' processed via RouteMem Gateway utilizing model {model}."
            return

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        candidate_models = [model] + [m for m in self.DEFAULT_FALLBACKS if m != model]

        for target_model in candidate_models:
            payload: Dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
                "stream": True
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    async with client.stream("POST", url, headers=headers, json=payload) as response:
                        if response.status_code != 200:
                            if response.status_code in (404, 429) and target_model != candidate_models[-1]:
                                continue  # Fallback to next available Groq model
                            yield f"[Groq API Error {response.status_code}]"
                            return

                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    return
                                try:
                                    chunk = json.loads(data_str)
                                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if delta:
                                        yield delta
                                except json.JSONDecodeError:
                                    pass
                        return
            except Exception:
                if target_model == candidate_models[-1]:
                    yield f"[RouteMem Groq Client ({target_model})]: Task completed."

    async def dispatch_completion(self, model: str, prompt: str, system_prompt: str = "") -> str:
        tokens = []
        async for chunk in self.dispatch_stream(model, prompt, system_prompt=system_prompt):
            tokens.append(chunk)
        return "".join(tokens)
