import json
from typing import AsyncGenerator, Optional
import httpx
from app.config import settings
from app.backends.base import BaseLLMBackend
from app.utils.logger import get_logger

logger = get_logger("vllm_client")

class VLLMClient(BaseLLMBackend):
    """Local SLM Client executing Ollama llama3.2:1b / vLLM workers locally."""

    DEFAULT_MAX_TOKENS = 1024
    MAX_SAFETY_CEILING = 4096

    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = endpoint_url or getattr(settings, "local_slm_url", "http://localhost:11434/v1")

    def _resolve_max_tokens(self, max_tokens: Optional[int]) -> int:
        """Resolves output token cap dynamically.
        
        1. If explicitly requested by caller, clamp to MAX_SAFETY_CEILING (4096).
        2. Otherwise, fall back to configured default (1024).
        """
        default_cap = getattr(settings, "local_slm_default_max_tokens", self.DEFAULT_MAX_TOKENS)
        ceiling = getattr(settings, "local_slm_max_tokens_ceiling", self.MAX_SAFETY_CEILING)
        if max_tokens is not None and max_tokens > 0:
            return min(max_tokens, ceiling)
        return default_cap

    async def dispatch_stream(self, model: str, prompt: str, system_prompt: str = "", max_tokens: Optional[int] = None) -> AsyncGenerator[str, None]:
        # Dynamic model resolver for installed local SLM catalog
        m_lower = model.lower() if model else ""
        if "specialist" in m_lower or "routemem" in m_lower:
            target_model = "routemem-specialist"
        elif "qwen" in m_lower or "code" in m_lower:
            target_model = "qwen2.5-coder:3b"
        elif "deepseek" in m_lower or "reason" in m_lower or "r1" in m_lower:
            target_model = "deepseek-r1:1.5b"
        elif "phi" in m_lower:
            target_model = "phi3.5:latest"
        elif "8b" in m_lower or "llama-3.1" in m_lower:
            target_model = "llama3.1:8b"
        else:
            target_model = "llama3.2:3b"

        ollama_url = "http://localhost:11434/v1/chat/completions"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resolved_tokens = self._resolve_max_tokens(max_tokens)
        temp = 0.2 if ("coder" in target_model or "deepseek" in target_model) else 0.7

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": True,
            "temperature": temp,
            "max_tokens": resolved_tokens,
            "options": {
                "temperature": temp,
                "num_predict": resolved_tokens,
                "stop": ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>", "\n\nUser:", "\n\nHuman:"]
            }
        }

        success = False
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", ollama_url, json=payload) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if delta:
                                        success = True
                                        yield delta
                                except json.JSONDecodeError:
                                    pass
                        if success:
                            return
        except Exception as e:
            logger.warning(f"Local Ollama SLM worker unreachable: {e}")

        # Fallback to Groq LPU SLM if local worker offline
        from app.backends.groq_client import GroqClient
        groq = GroqClient()
        async for chunk in groq.dispatch_stream("openai/gpt-oss-20b", prompt, system_prompt=system_prompt, max_tokens=max_tokens):
            yield chunk

    async def dispatch_completion(self, model: str, prompt: str, system_prompt: str = "", max_tokens: Optional[int] = None) -> str:
        tokens = []
        async for chunk in self.dispatch_stream(model, prompt, system_prompt=system_prompt, max_tokens=max_tokens):
            tokens.append(chunk)
        return "".join(tokens)
