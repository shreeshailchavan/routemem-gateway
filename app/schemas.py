import time
import uuid
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = Field(default="routemem-auto")
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    max_cost_target: Optional[float] = None
    quality_target: Optional[float] = 0.95

class RouteMemMetadata(BaseModel):
    cache_status: str  # EXACT_HIT, SEMANTIC_HIT, EXACT_MISS_SLM_HIT, GROQ_LPU_HIT, GEMINI_API_HIT
    ttft_ms: float
    latency_ms: float = 0.0
    confidence: float = 0.95
    token_reduction_ratio: float = 0.0
    cost_usd: float = 0.0
    routed_model: str = "routemem-auto"

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str

class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-routemem-{uuid.uuid4().hex[:10]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: UsageInfo
    routemem_metadata: RouteMemMetadata

    @classmethod
    def from_cache(
        cls,
        content: str,
        cache_status: str,
        ttft_ms: float,
        model_name: str = "routemem-cache"
    ) -> "ChatCompletionResponse":
        return cls(
            model=model_name,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=content),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=0,
                completion_tokens=len(content.split()),
                total_tokens=len(content.split())
            ),
            routemem_metadata=RouteMemMetadata(
                cache_status=cache_status,
                ttft_ms=round(ttft_ms, 2),
                token_reduction_ratio=0.0,
                cost_usd=0.0,
                routed_model=model_name
            )
        )
