# RouteMem AI Gateway — Master Development & Implementation Plan

This master plan details the exact file structure, module implementations, infrastructure orchestration, and testing plan for the **RouteMem AI Gateway**, designed for remote development on AWS EC2 (`t4g.xlarge` Control Plane + `g5.xlarge` GPU Worker) via VS Code Remote-SSH and AWS SSM.

---

## 1. Directory Blueprint & Target Files

```text
routemem-gateway/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI server & 8-stage pipeline orchestration
│   ├── config.py                   # Pydantic settings & environment configuration
│   ├── schemas.py                  # Pydantic request/response OpenAI-compatible schemas
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── exact_cache.py          # Tier-0 Redis SHA-256 exact hash cache (SLA: 2ms)
│   │   └── semantic_cache.py       # Tier-1 Qdrant HNSW vector search cache (SLA: <15ms)
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── zep_graphiti.py         # Zep temporal graph state integration
│   │   └── compressor.py           # LLMLingua-2 prompt token compressor (-80% tokens)
│   ├── router/
│   │   ├── __init__.py
│   │   ├── profiler.py             # ONNX DeBERTa difficulty & intent profiler (<3ms)
│   │   ├── uniroute.py             # Capability vector space mapper
│   │   └── omnirouter.py           # Lagrangian dual budget & quality solver (4ms)
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract LLM backend client interface
│   │   ├── sglang_client.py        # SGLang RadixAttention & LMCache client
│   │   ├── vllm_client.py          # Local vLLM worker client
│   │   └── cloud_client.py         # OpenAI / Anthropic cloud API client
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py              # Prometheus metrics collector & endpoint
│       └── logger.py               # Structured JSON logger
├── config/
│   ├── config.yaml                 # Gateway operational settings & SLAs
│   └── models.yaml                 # Model capability specs, latency, & pricing vectors
├── docker/
│   ├── Dockerfile                  # Gateway production Dockerfile
│   └── docker-compose.yml          # Full stack compose (Gateway, Redis, Qdrant)
├── tests/
│   ├── test_cache.py               # Tier-0 & Tier-1 unit test suite
│   ├── test_router.py              # Profiler & OmniRouter unit test suite
│   └── test_simulation.py          # 100-query benchmark & cost/TTFT verification
├── .env.example                    # Environment variable template
├── requirements.txt                # Python dependencies
└── README.md                       # Gateway setup & operational guide
```

---

## 2. Detailed Task Breakdown by Phase

### Phase 1: Core Scaffolding & Configuration Management
- [ ] **Directory Layout**: Create complete project folder structure.
- [ ] **Dependencies (`requirements.txt`)**:
  - `fastapi`, `uvicorn[standard]` (web server)
  - `pydantic`, `pydantic-settings`, `pyyaml` (config & data validation)
  - `redis[hiredis]` (Tier-0 cache)
  - `qdrant-client` (Tier-1 vector DB)
  - `onnxruntime`, `numpy` (Tier-3 ONNX profiler)
  - `httpx` (async HTTP backend client)
  - `prometheus-client` (metrics)
  - `pytest`, `pytest-asyncio` (testing framework)
- [ ] **Operational Configurations**:
  - `config/config.yaml`: Cache TTLs, semantic thresholds ($\ge 0.95$), compression target ratios (0.20), budget limits ($500/mo).
  - `config/models.yaml`: Pricing and capability specs for `llama-3.1-8b-instruct`, `qwen-2.5-coder-32b`, `claude-3.5-sonnet`, `gpt-4o`.
  - `app/config.py`: Pydantic settings loading `.env` and YAML config files.
- [ ] **API Schemas (`app/schemas.py`)**:
  - OpenAI-compatible `ChatCompletionRequest` & `ChatCompletionResponse`.
  - Extended `RouteMemMetadata` containing `cache_status`, `ttft_ms`, `compression_ratio`, `cost_usd`, `routed_model`.

---

### Phase 2: Tier-0 & Tier-1 Multi-Level Caching Subsystem
- [ ] **Tier-0 Exact Cache (`app/cache/exact_cache.py`)**:
  - Async Redis connection manager.
  - SHA-256 key hashing: `key = "exact_cache:" + SHA256(system_prompt || "::" || user_prompt)`.
  - Get/Set methods with configurable TTL (default 86400s / 24h).
- [ ] **Tier-1 Semantic Cache (`app/cache/semantic_cache.py`)**:
  - Async Qdrant client connection manager.
  - Query vectorizer using `bge-small-en-v1.5` (384 dimensions).
  - Cosine distance similarity search with configurable threshold ($\ge 0.95$).
  - Vector indexing on background cache update.

---

### Phase 3: Tier-2 Shared Memory & Token Compression
- [ ] **Context Compressor (`app/memory/compressor.py`)**:
  - LLMLingua-2 prompt compressor integration.
  - Prunes low-entropy prompt tokens while preserving structural syntax (target ~80% reduction).
- [ ] **Zep Temporal Memory Integration (`app/memory/zep_graphiti.py`)**:
  - Client interface for Zep graph memory service.
  - Extracts and updates session entity facts across multi-turn user requests.

---

### Phase 4: Tier-3 Capability-Aware Dynamic Router
- [ ] **Query Profiler (`app/router/profiler.py`)**:
  - ONNX Runtime execution engine loading quantized `DeBERTa-v3`.
  - Structural code AST detection, math symbol density analysis, and semantic difficulty evaluation ($D \in [0.0, 1.0]$).
  - Intent classification (e.g., `code_generation`, `simple_qa`, `complex_reasoning`).
- [ ] **Capability Mapper (`app/router/uniroute.py`)**:
  - Maps model performance vectors into a unified capability space.
- [ ] **Lagrangian Solver (`app/router/omnirouter.py`)**:
  - Solves dual loss: $L = \text{Cost} - \lambda \cdot (\text{Predicted\_Acc} - \alpha_{\text{target}})$.
  - Selects minimum-cost backend that satisfies user quality constraint $\alpha_{\text{target}}$.

---

### Phase 5: Tier-4 Backend Dispatchers & Execution Clients
- [ ] **Abstract Backend Base (`app/backends/base.py`)**:
  - Abstract base class for non-blocking async LLM streaming.
- [ ] **Local vLLM Worker Client (`app/backends/vllm_client.py`)**:
  - Async client communicating with vLLM OpenAI-compatible API server on port 8001.
  - Enables prefix caching and stream parsing.
- [ ] **SGLang Client (`app/backends/sglang_client.py`)**:
  - Client for SGLang with RadixAttention KV cache reuse & LMCache host RAM/NVMe offloading.
- [ ] **Cloud API Fallback Client (`app/backends/cloud_client.py`)**:
  - Async streaming client for OpenAI (GPT-4o, GPT-4o-mini) and Anthropic (Claude 3.5 Sonnet, Haiku).

---

### Phase 6: Gateway Server & Non-Blocking Async State Sync
- [ ] **FastAPI Core (`app/main.py`)**:
  - `/v1/chat/completions` endpoint orchestrating the 8-stage pipeline.
  - Non-blocking background state sync (`asyncio.create_task(sync_background_state)`) updating Redis, Qdrant, and Zep without adding latency to the response stream.
  - Health check endpoint `/health`.
- [ ] **Observability (`app/utils/logger.py` & `app/utils/metrics.py`)**:
  - Structured JSON logging.
  - Prometheus metrics exporter tracking cache hit rates, TTFT, token compression ratio, and model spend.

---

### Phase 7: Containerization & AWS EC2 Provisioning Artifacts
- [ ] **Gateway Dockerfile (`docker/Dockerfile`)**: Multi-stage Python container build.
- [ ] **Orchestration (`docker/docker-compose.yml`)**: Links Gateway, Redis 7, and Qdrant v1.7.4.
- [ ] **AWS Setup Script & README (`README.md`)**:
  - AWS EC2 `t4g.xlarge` (Control Plane) and `g5.xlarge` (GPU Worker Spot) provisioning instructions.
  - VS Code Remote-SSH configuration guide.

---

### Phase 8: Benchmarking & Test Suite
- [ ] **Unit Tests (`tests/test_cache.py` & `tests/test_router.py`)**:
  - Validates exact cache hit/miss logic and TTL.
  - Validates Qdrant HNSW vector thresholding ($\ge 0.95$).
  - Validates Profiler difficulty scoring and OmniRouter model selection.
- [ ] **Simulation & 100-Query Benchmark (`tests/test_simulation.py`)**:
  - Runs 100 benchmark queries.
  - Verifies SLAs: exact cache TTFT `<2ms`, semantic cache TTFT `<15ms`, token reduction `>80%`, spend drop `>95%`.

---

## 3. Execution Strategy Overview

```
[Phase 1: Scaffolding & Config] -> [Phase 2: Redis & Qdrant Cache] -> [Phase 3: LLMLingua-2 & Zep]
                                                                                │
                                                                                ▼
[Phase 6: Gateway & Async Sync] <- [Phase 5: vLLM & Cloud Backends] <- [Phase 4: ONNX Profiler & OmniRouter]
             │
             ▼
[Phase 7: Docker & AWS Setup] -> [Phase 8: 100-Query Benchmark Verification]
```
