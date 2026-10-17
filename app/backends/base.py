from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any, Optional

def build_chat_messages(
    prompt: str,
    system_prompt: str = "",
    conversation_history: Optional[List[Any]] = None
) -> List[Dict[str, str]]:
    """Reconstructs full multi-turn messages array for LLM backends.
    
    Includes:
    1. System prompt (enriched with retrieved KG facts / intent instructions)
    2. Prior conversational turns (user/assistant) from conversation_history
    3. Final user query (prompt)
    """
    messages: List[Dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    if conversation_history and len(conversation_history) > 1:
        # Preceding dialogue turns, excluding the last prompt which is passed as prompt
        for m in conversation_history[:-1]:
            role = getattr(m, "role", "") if not isinstance(m, dict) else m.get("role", "")
            content = getattr(m, "content", "") if not isinstance(m, dict) else m.get("content", "")
            if role in ("user", "assistant") and content and content.strip():
                messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": prompt})
    return messages

class BaseLLMBackend(ABC):
    """Abstract Base Class for LLM Client Backends."""

    @abstractmethod
    async def dispatch_stream(self, model: str, prompt: str, system_prompt: str = "", **kwargs) -> AsyncGenerator[str, None]:
        """Stream completion text tokens from backend."""
        pass

    @abstractmethod
    async def dispatch_completion(self, model: str, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Return complete response text from backend."""
        pass
