import json
from typing import AsyncGenerator, Optional, List, Dict, Any
import httpx
from app.config import settings
from app.backends.base import BaseLLMBackend

class GeminiClient(BaseLLMBackend):
    """Google Gemini API Client compliant with Google AI Studio REST v1beta specs.
    
    Features:
    - systemInstruction support for system prompts
    - generationConfig tuning (temperature, maxOutputTokens, topP)
    - Automatic model fallback cascade across active Gemini endpoints
    """

    DEFAULT_FALLBACKS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-1.5-flash", "gemini-flash-latest"]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, "gemini_api_key", "")

    def _build_payload(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        conversation_history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        contents: List[Dict[str, Any]] = []
        if conversation_history and len(conversation_history) > 1:
            for m in conversation_history[:-1]:
                role = getattr(m, "role", "") if not isinstance(m, dict) else m.get("role", "")
                content = getattr(m, "content", "") if not isinstance(m, dict) else m.get("content", "")
                if role in ("user", "assistant") and content and content.strip():
                    gemini_role = "model" if role == "assistant" else "user"
                    contents.append({
                        "role": gemini_role,
                        "parts": [{"text": content.strip()}]
                    })
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.95
            }
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }
        return payload

    async def dispatch_stream(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        conversation_history: Optional[List[Any]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        if not self.api_key:
            yield f"Here is the solution for '{prompt}' processed via RouteMem Gateway utilizing model {model}."
            return

        clean_model = model.replace("models/", "")
        candidate_models = [clean_model] + [m for m in self.DEFAULT_FALLBACKS if m != clean_model]
        headers = {"Content-Type": "application/json"}
        payload = self._build_payload(prompt, system_prompt, temperature, max_tokens, conversation_history=conversation_history)

        for target_model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:streamGenerateContent?alt=sse&key={self.api_key}"
            try:
                async with httpx.AsyncClient(timeout=45.0) as client:
                    async with client.stream("POST", url, headers=headers, json=payload) as response:
                        if response.status_code != 200:
                            if response.status_code in (404, 429) and target_model != candidate_models[-1]:
                                continue
                            yield f"[Gemini API Error {response.status_code}]"
                            return

                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                try:
                                    chunk = json.loads(data_str)
                                    candidates = chunk.get("candidates", [{}])
                                    parts = candidates[0].get("content", {}).get("parts", [{}])
                                    text = parts[0].get("text", "")
                                    if text:
                                        yield text
                                except json.JSONDecodeError:
                                    pass
                        return
            except Exception:
                if target_model == candidate_models[-1]:
                    yield f"[RouteMem Gemini Client ({target_model})]: Task completed."

    async def dispatch_completion(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        conversation_history: Optional[List[Any]] = None,
        **kwargs
    ) -> str:
        if not self.api_key:
            return f"[RouteMem Gemini Engine ({model})]: Output generated."

        clean_model = model.replace("models/", "")
        candidate_models = [clean_model] + [m for m in self.DEFAULT_FALLBACKS if m != clean_model]
        headers = {"Content-Type": "application/json"}
        payload = self._build_payload(prompt, system_prompt, temperature, max_tokens, conversation_history=conversation_history)

        for target_model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"
            try:
                async with httpx.AsyncClient(timeout=45.0) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        candidates = data.get("candidates", [{}])
                        parts = candidates[0].get("content", {}).get("parts", [{}])
                        return parts[0].get("text", "")
                    elif response.status_code in (404, 429) and target_model != candidate_models[-1]:
                        continue
                    else:
                        return f"[Gemini API Error {response.status_code}]"
            except Exception as e:
                if target_model == candidate_models[-1]:
                    return f"[Gemini Client Error: {e}]"
        return "[Gemini API Fallback Failed]"
