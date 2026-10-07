from abc import ABC, abstractmethod
from typing import AsyncGenerator

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
