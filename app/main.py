import time
import json
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, Response, JSONResponse

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    def generate_latest(): return b"# Prometheus metrics exporter ready.\n"
    CONTENT_TYPE_LATEST = "text/plain"

from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    RouteMemMetadata,
    Choice,
    ChoiceMessage,
    UsageInfo
)
from app.cache.exact_cache import ExactCache
from app.cache.semantic_cache import SemanticCache
from app.memory.compressor import PromptCompressor
from app.memory.zep_graphiti import ZepGraphitiMemory
from app.router.profiler import QueryProfiler
from app.router.omnirouter import OmniRouter
from app.backends.vllm_client import VLLMClient
from app.backends.cloud_client import CloudAPIClient
from app.backends.groq_client import GroqClient
from app.backends.deepseek_client import DeepSeekClient
from app.backends.gemini_client import GeminiClient
from app.utils.logger import get_logger
from app.utils.metrics import REQUEST_COUNT, CACHE_HITS, TTFT_HISTOGRAM, TOKEN_REDUCTION_GAUGE
from app.config import settings

logger = get_logger("routemem_gateway")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="RouteMem AI Gateway",
    description="Enterprise Multi-LLM Proxy Gateway with 4-Tier Memory & Dynamic Routing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service Initializations
exact_cache = ExactCache()
semantic_cache = SemanticCache()
compressor = PromptCompressor()
zep_memory = ZepGraphitiMemory()
profiler = QueryProfiler()
router = OmniRouter()

vllm_client = VLLMClient()
cloud_client = CloudAPIClient()
groq_client = GroqClient()
deepseek_client = DeepSeekClient()
gemini_client = GeminiClient()

async def sync_background_state(system_prompt: str, user_prompt: str, response_text: str, session_id: str = "default-session"):
    """Stage 8: Non-blocking background state sync across Redis, Qdrant & Zep Graphiti."""
    try:
        await exact_cache.set(system_prompt, user_prompt, response_text)
        await semantic_cache.index(user_prompt, response_text)
        await zep_memory.add_session_interaction(session_id, user_prompt, response_text)
    except Exception as e:
        logger.error(f"Background state sync error: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RouteMem AI Gateway", "version": "1.0.0"}

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    if isinstance(data, str):
        data = data.encode("utf-8")
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/admin/cache/clear")
@app.get("/v1/admin/cache/clear")
async def clear_all_caches():
    """Flushes Redis exact cache, Qdrant semantic vector cache, and Zep Graphiti KG memory."""
    redis_cleared = await exact_cache.clear()
    qdrant_cleared = await semantic_cache.clear()
    zep_cleared = await zep_memory.clear()
    return {
        "status": "success",
        "message": "All memory tiers successfully cleared.",
        "details": {
            "tier_0_redis_exact_cache": "cleared" if redis_cleared else "skipped/offline",
            "tier_1_qdrant_semantic_cache": "cleared" if qdrant_cleared else "skipped/offline",
            "tier_2_zep_graphiti_knowledge_graph": "cleared" if zep_cleared else "skipped/offline"
        }
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    start_time = time.perf_counter()
    REQUEST_COUNT.labels(method="POST", endpoint="/v1/chat/completions", status="200").inc()

    # Stage 1: Ingestion & Extraction
    session_id = request.session_id or "default-session"
    system_prompt = next((m.content for m in request.messages if m.role == "system"), "")
    user_messages = [m.content for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided in request.")
    user_prompt = user_messages[-1]

    # Stage 2: Tier-0 Exact Hash Cache Check (2ms)
    exact_hit = await exact_cache.get(system_prompt, user_prompt)
    if exact_hit:
        ttft_ms = (time.perf_counter() - start_time) * 1000
        CACHE_HITS.labels(cache_type="exact").inc()
        TTFT_HISTOGRAM.observe(ttft_ms / 1000.0)
        return ChatCompletionResponse.from_cache(
            content=exact_hit,
            cache_status="EXACT_HIT",
            ttft_ms=ttft_ms,
            model_name="redis-exact-hash-cache"
        )

    # Stage 3: Tier-1 Semantic Vector Cache Check (<15ms)
    semantic_hit = await semantic_cache.search(user_prompt, threshold=settings.semantic_cache_threshold)
    if semantic_hit:
        ttft_ms = (time.perf_counter() - start_time) * 1000
        CACHE_HITS.labels(cache_type="semantic").inc()
        TTFT_HISTOGRAM.observe(ttft_ms / 1000.0)
        return ChatCompletionResponse.from_cache(
            content=semantic_hit.response,
            cache_status="SEMANTIC_HIT",
            ttft_ms=ttft_ms,
            model_name="qdrant-semantic-vector-cache"
        )

    # Stage 4: Shared Memory Context & Token Compression (5ms)
    compressed_prompt, token_reduction_ratio = compressor.compress(user_prompt)
    TOKEN_REDUCTION_GAUGE.set(token_reduction_ratio)

    session_facts = await zep_memory.get_session_context(session_id)
    kg_facts_retrieved_count = 0
    kg_memory_used = False
    if session_facts:
        kg_facts_retrieved_count = len([f for f in session_facts.split('\n') if f.strip()])
        kg_memory_used = True
        system_prompt = f"{system_prompt}\n\n[Retrieved Session Knowledge Graph Memory]:\n{session_facts}".strip()

    # Stage 5: Intent & Difficulty Profiling (<3ms)
    difficulty_score, task_intent = profiler.profile(compressed_prompt)

    # Stage 6: Capability Space & Budget Optimization (4ms)
    if request.model and request.model not in ["routemem-auto", "auto", "default"]:
        selected_model = request.model
    else:
        selected_model = router.select_model(
            difficulty=difficulty_score,
            intent=task_intent,
            max_cost_target=request.max_cost_target,
            quality_target=request.quality_target
        )

    # Stage 7: Dispatch to Selected Model Backend
    target_routed_model = selected_model
    actual_answering_model = selected_model
    cache_status = "EXACT_MISS_ROUTED"

    local_keywords = ["local", "llama", "phi", "mistral", "qwen2.5-coder", "deepseek-r1:1.5b", "ollama"]
    if any(k in selected_model.lower() for k in local_keywords):
        backend_client = vllm_client
        cache_status = "LOCAL_SLM_HIT"
        actual_answering_model = selected_model
    elif "groq" in selected_model or "deepseek" in selected_model:
        backend_client = groq_client
        cache_status = "GROQ_LPU_HIT"
        if "qwen" in selected_model or "coder" in selected_model:
            actual_answering_model = "qwen/qwen3.8-27b"
        elif "20b" in selected_model or "8b" in selected_model:
            actual_answering_model = "openai/gpt-oss-20b"
        else:
            actual_answering_model = "openai/gpt-oss-120b"
    elif "gemini" in selected_model:
        backend_client = gemini_client
        cache_status = "GEMINI_API_HIT"
        actual_answering_model = "gemini-3.8-flash"
    elif "claude" in selected_model or "gpt" in selected_model or "o1" in selected_model or "o3" in selected_model:
        backend_client = cloud_client
        cache_status = "CLOUD_FALLBACK"
        actual_answering_model = "openai/gpt-oss-120b"
    else:
        backend_client = vllm_client
        cache_status = "LOCAL_SLM_HIT"
        actual_answering_model = selected_model

    actual_model_name = actual_answering_model

    if request.stream:
        async def stream_generator():
            full_response = []
            first_token_time = None
            async for token in backend_client.dispatch_stream(actual_model_name, compressed_prompt, system_prompt=system_prompt):
                if first_token_time is None:
                    first_token_time = (time.perf_counter() - start_time) * 1000
                    TTFT_HISTOGRAM.observe(first_token_time / 1000.0)
                full_response.append(token)
                chunk_data = json.dumps({"choices": [{"delta": {"content": token}}], "model": actual_model_name})
                yield f"data: {chunk_data}\n\n"

            ttft_ms = round(first_token_time or ((time.perf_counter() - start_time) * 1000), 2)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            meta_chunk = json.dumps({
                "routemem_metadata": {
                    "cache_status": cache_status,
                    "ttft_ms": ttft_ms,
                    "latency_ms": latency_ms,
                    "confidence": round(1.0 - (difficulty_score * 0.25), 2),
                    "token_reduction_ratio": token_reduction_ratio,
                    "cost_usd": 0.000000 if "LOCAL" in cache_status or "CACHE" in cache_status else 0.000002,
                    "target_routed_model": target_routed_model,
                    "actual_answering_model": actual_model_name,
                    "routed_model": actual_model_name,
                    "kg_facts_retrieved": kg_facts_retrieved_count,
                    "kg_memory_used": kg_memory_used,
                    "is_fallback": target_routed_model != actual_model_name
                }
            })
            yield f"data: {meta_chunk}\n\n"
            yield "data: [DONE]\n\n"

            # Stage 8: Async Background Sync
            final_text = "".join(full_response)
            asyncio.create_task(sync_background_state(system_prompt, user_prompt, final_text, session_id=session_id))

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Non-streaming response path
    full_response = await backend_client.dispatch_completion(actual_model_name, compressed_prompt, system_prompt=system_prompt)
    ttft_ms = (time.perf_counter() - start_time) * 1000
    TTFT_HISTOGRAM.observe(ttft_ms / 1000.0)

    # Stage 8: Async Background Sync (0ms blocking)
    asyncio.create_task(sync_background_state(system_prompt, user_prompt, full_response, session_id=session_id))

    return ChatCompletionResponse(
        model=actual_model_name,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=full_response),
                finish_reason="stop"
            )
        ],
        usage=UsageInfo(
            prompt_tokens=len(compressed_prompt.split()),
            completion_tokens=len(full_response.split()),
            total_tokens=len(compressed_prompt.split()) + len(full_response.split())
        ),
        routemem_metadata=RouteMemMetadata(
            cache_status=cache_status,
            ttft_ms=round(ttft_ms, 2),
            latency_ms=round(ttft_ms, 2),
            confidence=round(1.0 - (difficulty_score * 0.25), 2),
            token_reduction_ratio=token_reduction_ratio,
            cost_usd=0.000000 if "LOCAL" in cache_status or "CACHE" in cache_status else 0.000002,
            target_routed_model=target_routed_model,
            actual_answering_model=actual_model_name,
            routed_model=actual_model_name,
            kg_facts_retrieved=kg_facts_retrieved_count,
            kg_memory_used=kg_memory_used,
            is_fallback=target_routed_model != actual_model_name
        )
    )
