# RouteMem AI Gateway — Gap Analysis & Production Review

This document provides a gap analysis comparing the current codebase against the specifications in `routemem_dev_spec.md` and `routemem_aws_deployment_guide.md`.

---

## 1. Executive Summary of Audit

| Subsystem / Feature | Current Implementation Status | Gap / Needed Production Hardening | Priority |
| :--- | :--- | :--- | :--- |
| **Tier-0 Exact Cache** | Fully Implemented (`app/cache/exact_cache.py`) | Production-ready with SHA-256 Redis TTL | Completed |
| **Tier-1 Semantic Cache** | Functional Vectorizer Fallback (`app/cache/semantic_cache.py`) | Needs ONNX / `fastembed` `bge-small-en-v1.5` dense embedding model integration | High |
| **Tier-2 Memory & Token Pruning** | Functional (`app/memory/compressor.py` & `zep_graphiti.py`) | Zep `ZEP_API_KEY` header support & graphiti entity extraction payload formatting | Medium |
| **Tier-3 Profiler & Router** | Fully Functional (`app/router/profiler.py` & `omnirouter.py`) | Needs `models/download_onnx.py` script to fetch quantized `DeBERTa-v3` weights | Medium |
| **Tier-4 Model Backends** | Fully Functional (`vllm_client`, `sglang_client`, `cloud_client`) | Needs explicit `LMCache` host RAM offload headers (`x-lmcache-enable`) | Medium |
| **FastAPI Control Plane** | Fully Functional (`app/main.py`) | Needs HTTP middleware for auto-recording Prometheus metrics on all endpoints | Medium |
| **AWS Infrastructure Setup** | Documented in `README.md` & Deployment Guide | Needs automated `scripts/setup_aws_control_plane.sh` & `scripts/aws_autostop_alarm.sh` | High |
| **Integration Test Suite** | Simulation & Unit Tests Done (`test_router.py`, `test_simulation.py`) | Needs `tests/test_api.py` using `TestClient` for `/v1/chat/completions` endpoint | Medium |

---

## 2. Detailed Breakdown of Missing / Enhancement Items

### Item 1: Real Dense Embeddings for Tier-1 Semantic Cache (`bge-small-en-v1.5`)
- **Current State**: Uses a deterministic 384-dim hashing vectorizer as a fallback when HuggingFace transformer weights are not downloaded locally.
- **Enhancement Needed**: Add `fastembed` / `onnxruntime` native `bge-small-en-v1.5` ONNX model loader in `app/cache/semantic_cache.py` so semantic similarity is evaluated using true neural embeddings.

### Item 2: ONNX Model Downloader Script (`scripts/download_onnx_models.py`)
- **Current State**: `QueryProfiler` is prepared to load `models/deberta_v3_profiler.onnx`.
- **Enhancement Needed**: Provide an automated download script `scripts/download_onnx_models.py` to pull down quantized `DeBERTa-v3` ONNX model weights and tokenizer configs.

### Item 3: Prometheus Auto-Metrics FastAPI Middleware
- **Current State**: `app/utils/metrics.py` defines Prometheus counters and histograms, and `app/main.py` exposes `/metrics`.
- **Enhancement Needed**: Register a FastAPI middleware in `app/main.py` to auto-record latency, TTFT, status codes, and request counts across all API calls.

### Item 4: AWS Infrastructure Setup & Auto-Stop Cost Alarm Scripts (`scripts/`)
- **Current State**: Manual deployment steps documented in `routemem_aws_deployment_guide.md`.
- **Enhancement Needed**:
  - `scripts/setup_control_plane.sh`: Installs Docker, Redis, Qdrant, Python env, and systemd service on EC2 `t4g.xlarge`.
  - `scripts/setup_gpu_worker.sh`: Launches vLLM OpenAI server on Spot EC2 `g5.xlarge`.
  - `scripts/aws_cloudwatch_autostop.sh`: Configures AWS CloudWatch alarm to automatically stop Spot GPU workers if GPU/CPU utilization drops below 5% for 30 minutes.

### Item 5: End-to-End FastAPI Integration Test (`tests/test_api.py`)
- **Current State**: `test_router.py`, `test_cache.py`, and `test_simulation.py` exist.
- **Enhancement Needed**: Add `tests/test_api.py` using `fastapi.testclient.TestClient` to test `/v1/chat/completions` endpoint, streaming SSE responses, `/health`, and `/metrics`.

---

## 3. Recommended Remediation Plan

We can immediately build out these 5 remaining production enhancement items to make the project **100% complete and production-hardened**:

1. Create `app/cache/dense_embedder.py` adding `bge-small-en-v1.5` neural embedding support to Tier-1 Semantic Cache.
2. Add FastAPI Prometheus Middleware to `app/main.py`.
3. Create `scripts/download_onnx_models.py` for downloading quantized ONNX profiler weights.
4. Create AWS automation scripts: `scripts/setup_control_plane.sh`, `scripts/setup_gpu_worker.sh`, and `scripts/aws_autostop_alarm.sh`.
5. Create `tests/test_api.py` for end-to-end FastAPI endpoint integration testing.
