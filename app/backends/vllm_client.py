import json
from typing import AsyncGenerator, Optional
import httpx
from app.config import settings
from app.backends.base import BaseLLMBackend
from app.utils.logger import get_logger

logger = get_logger("vllm_client")

class VLLMClient(BaseLLMBackend):
    """Local SLM Client executing Ollama llama3.2:1b / vLLM workers locally."""

    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = endpoint_url or getattr(settings, "local_slm_url", "http://localhost:11434/v1")

    async def dispatch_stream(self, model: str, prompt: str, system_prompt: str = "") -> AsyncGenerator[str, None]:
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

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7
        }

        success = False
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
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
        async for chunk in groq.dispatch_stream("openai/gpt-oss-20b", prompt, system_prompt=system_prompt):
            yield chunk

    async def dispatch_completion(self, model: str, prompt: str, system_prompt: str = "") -> str:
        tokens = []
        async for chunk in self.dispatch_stream(model, prompt, system_prompt=system_prompt):
            tokens.append(chunk)
        return "".join(tokens)
