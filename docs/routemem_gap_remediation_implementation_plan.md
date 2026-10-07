# RouteMem AI Gateway: Comprehensive Gap Remediation & Implementation Plan

**Document Version:** 1.0.0  
**Status:** Under Architecture Review  
**Target Environment:** RouteMem Control Plane (AWS EC2 & Local Repository)  
**Date:** October 7, 2026  

---

## 1. Executive Summary & Objective

Following our comprehensive gap analysis between the theoretical foundations (RouteLLM, UniRoute, Graphiti, SGLang, and OmniRouter) and the current production implementation, this document establishes a **rigorous, step-by-step remediation plan** for each identified gap.

Every remediation initiative is detailed with:
1. **The Core Problem & Root Cause**
2. **Proposed Solution & System Architecture**
3. **Step-by-Step Implementation Sequence**
4. **Technical & Scientific Justification**
5. **Expected Quantitative Impact & Metric Targets**
6. **Risks & Mitigation Strategies**

---

## 2. Initiative 1: RouteLLM Neural Preference Head ONNX Integration (Gap 1.1)

### 2.1 The Problem & Root Cause
- **Current State:** The RouteLLM pairwise preference head ([`models/preference_head.pt`](file:///home/monarch/shreeshail/dev/personal/projects/routemem/models/preference_head.pt)) was trained via PyTorch ([`scripts/train_routellm_head.py`](file:///home/monarch/shreeshail/dev/personal/projects/routemem/scripts/train_routellm_head.py)) achieving $92.0\%$ accuracy. However, in the live request path ([`app/main.py`](file:///home/monarch/shreeshail/dev/personal/projects/routemem/app/main.py)), model routing relies on `profiler.py` and rule-based thresholds.
- **Root Cause:** Loading full `torch` libraries inside the FastAPI latency-critical request thread introduces $40 - 80\text{ ms}$ of memory overhead and heavy CPU footprint on ARM EC2 instances.

### 2.2 Proposed Solution & Architecture
Export the 3-layer PyTorch MLP weights to an optimized **INT8/FP32 ONNX runtime model** (`models/preference_head.onnx`). Integrate ONNX Runtime inference directly into `OmniRouter` (`app/router/omnirouter.py`):
$$\delta(M, q) = \text{ONNXRuntime}(\phi(q), \vec{c}_m)$$
$$P(\text{Local SLM Wins} \mid q) = \sigma(\delta_{\text{local}} - \delta_{\text{cloud}})$$

```
Incoming Query (q) ──► FastEmbed / ONNX Encoder ──► preference_head.onnx (< 0.2ms) ──► Win Prob P(wins) ──► OmniRouter Dual Solver
```

### 2.3 Step-by-Step Implementation Sequence
1. Create `scripts/export_routellm_onnx.py` using `torch.onnx.export` to serialize the PyTorch preference head.
2. Verify output numerical equivalence between PyTorch and ONNX runtime across 500 test vectors (tolerance $< 10^{-5}$).
3. Update `app/router/omnirouter.py` to initialize `onnxruntime.InferenceSession` on startup.
4. Replace static difficulty thresholds in `select_model()` with the continuous win probability $P(\text{wins} \mid q)$.

### 2.4 Justification & Metrics
- **Justification:** ONNX Runtime executes in pure C++ on CPU without importing PyTorch, eliminating the 80ms latency penalty while giving RouteMem true neural preference routing.
- **Target Metrics:** Routing decision execution time **$< 0.25\text{ ms}$**; pairwise routing precision **$\ge 92.0\%$**.

---

## 3. Initiative 2: Multi-Turn Contextual Vector Caching (Gap 1.3)

### 3.1 The Problem & Root Cause
- **Current State:** Tier-1 Semantic Vector Cache (`app/cache/semantic_cache.py`) embeds only the latest user string $q$ into Qdrant vector space.
- **Root Cause:** In multi-turn dialogues, queries like *"Can you explain that in more detail?"* or *"What was the first step again?"* lack semantic grounding on their own. They can falsely match unrelated queries from previous sessions, causing incorrect cache hits.

### 3.2 Proposed Solution & Architecture
Construct a **Composite Contextual Query Representation** before embedding:
$$q_{\text{composite}} = \text{SystemPrompt}_{[:128]} \parallel \text{" \| Context: "} \parallel \text{LastAssistantTurn}_{[:256]} \parallel \text{" \| Query: "} \parallel q_{\text{current}}$$
Additionally, store `session_id` and `turn_id` in Qdrant point payloads to support session-isolated or global similarity matching.

```
Multi-Turn Chat ──► Context Concatenator ──► FastEmbed (BGE 384-d) ──► Qdrant HNSW Search (τ=0.85) ──► Semantic Cache Hit/Miss
```

### 3.3 Step-by-Step Implementation Sequence
1. In `app/main.py`, extract the immediate preceding assistant response from the conversation message history.
2. In `app/cache/semantic_cache.py`, update `search()` and `index()` to accept an optional `context: Optional[str] = None`.
3. Concatenate context using a standardized delimiter before generating the 384-dim BGE embedding.
4. Add integration unit tests in `tests/test_cache.py` verifying that ambiguous queries in different sessions do not collide.

### 3.4 Justification & Metrics
- **Justification:** Context-aware semantic embedding is the gold standard in production RAG systems (SmartCache, Wang et al. NeurIPS 2025). It guarantees that conversational context is preserved in vector space.
- **Target Metrics:** Zero false positive cross-session semantic collisions; cache hit accuracy **$> 99.0\%$** on multi-turn conversations.

---

## 4. Initiative 3: Durable Disk Persistence for Zep Graphiti Knowledge Graph (Gap 1.4)

### 4.1 The Problem & Root Cause
- **Current State:** `ZepGraphitiMemory` (`app/memory/zep_graphiti.py`) attempts connection to Zep on port 8080, falling back to an in-memory dictionary `self._memory_store`.
- **Root Cause:** If the Uvicorn gateway process or EC2 instance reboots, all in-memory conversational entity facts are wiped clean, causing memory amnesia.

### 4.2 Proposed Solution & Architecture
Implement an **Embedded SQLite / File-Backed Write-Ahead Store** (`data/graphiti_memory.db`) with automatic persistence and thread-safe WAL (Write-Ahead Logging):
- Table `entities`: `(session_id, entity_name, entity_type, created_at, updated_at)`
- Table `relationships`: `(session_id, source_entity, relation, target_entity, temporal_weight)`

```
Session Dialogue ──► Entity Extractor ──► SQLite WAL Store (data/graphiti_memory.db) ──► Instant Recovery on Restart
```

### 4.3 Step-by-Step Implementation Sequence
1. Create `app/memory/sqlite_graph_store.py` providing an asynchronous, thread-safe SQLite persistence engine.
2. Refactor `app/memory/zep_graphiti.py` to use `sqlite_graph_store` as the permanent local backing store whenever remote Zep REST API is unreachable.
3. Add automatic index creation on `session_id` to guarantee sub-millisecond graph query times.
4. Verify that restarting Uvicorn preserves all previously learned session facts.

### 4.4 Justification & Metrics
- **Justification:** SQLite requires zero external containers, operates directly on disk with microsecond read latencies, and guarantees ACID compliance with zero memory leaks.
- **Target Metrics:** Graph fact retrieval latency **$< 1.5\text{ ms}$**; $100\%$ session fact retention across server restarts.

---

## 5. Initiative 4: Cognitive SLM Upgrade on ARM CPU (Gap 2.2)

### 5.1 The Problem & Root Cause
- **Current State:** Ollama on EC2 serves `llama3.2:1b`.
- **Root Cause:** A 1-billion parameter model has limited reasoning and code synthesis depth. Any request with difficulty $D > 0.45$ must escalate to Groq LPUs or Cloud models, reducing the proportion of queries served locally.

### 5.2 Proposed Solution & Architecture
Pull a highly capable quantized 3B model into Ollama on EC2:
- Primary Candidate: **`llama3.2:3b`** (Meta's flagship compact model) or **`qwen2.5:3b-instruct`** (Alibaba's top-performing 3B model for code & math).
- The `t4g.xlarge` instance has **16 GB RAM** and 4 Neoverse-N1 ARM cores, which comfortably runs a 4-bit quantized 3B model at **$18 - 25\text{ tokens/sec}$**.

### 5.3 Step-by-Step Implementation Sequence
1. SSH into EC2 and execute `ollama pull llama3.2:3b`.
2. Update `app/backends/vllm_client.py` to route local requests to `llama3.2:3b` as default local SLM.
3. Update `config/models.yaml` capability vector and benchmark expectations.
4. Benchmark TTFT and token generation speed on EC2 ARM CPU.

### 5.4 Justification & Metrics
- **Justification:** `llama3.2:3b` achieves **63.4% on MMLU** and **65.0% on GSM8K** (vs. ~38% on 1B), allowing RouteMem to comfortably handle queries up to difficulty $D = 0.60$ locally at $\$0.00$ cost.
- **Target Metrics:** Increase local query resolution ratio from $55\%$ to **$72\%$**, decreasing cloud API dependency.

---

## 6. Initiative 5: Production Process Supervision via Systemd (Gap 3.2)

### 6.1 The Problem & Root Cause
- **Current State:** Uvicorn is launched manually via `nohup` over SSH.
- **Root Cause:** If the EC2 instance reboots or crashes, Uvicorn remains down until manually restarted.

### 6.2 Proposed Solution & Architecture
Deploy a standard Linux systemd service unit: `/etc/systemd/system/routemem-gateway.service`
```ini
[Unit]
Description=RouteMem AI Gateway Service
After=network.target docker.service ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/routemem-gateway
Environment=PATH=/home/ubuntu/routemem-gateway/.venv/bin:/usr/local/bin:/usr/bin
ExecStart=/home/ubuntu/routemem-gateway/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### 6.3 Step-by-Step Implementation Sequence
1. Create `scripts/setup_systemd_service.sh` to install and enable the service unit file on EC2.
2. Execute `sudo systemctl daemon-reload && sudo systemctl enable --now routemem-gateway.service`.
3. Verify status with `systemctl status routemem-gateway.service`.

### 6.4 Justification & Metrics
- **Justification:** Standard Linux operational best practice ensuring high availability, automatic crash recovery, and instant restart on server reboots.
- **Target Metrics:** Zero-downtime automatic service recovery in **$< 3\text{ seconds}$**.

---

## 7. Initiative 6: Permanent Static IP & Endpoint Auto-Configuration (Gap 3.1)

### 7.1 The Problem & Root Cause
- **Current State:** Stopping and starting EC2 changes its public IP, requiring manual updates in client scripts.
- **Root Cause:** No AWS Elastic IP (EIP) or DNS hostname is attached to the instance.

### 7.2 Proposed Solution & Architecture
- **Option A (AWS Elastic IP):** Allocate an EIP via AWS CLI and associate it with instance `i-0720d9efd39dc72a8`. (Costs $\$0.00$ while instance is running).
- **Option B (Dynamic Endpoint Discovery Script):** If AWS CLI credentials expire or EIP quota is restricted, provide `scripts/sync_ec2_ip.sh` that detects the active public IP from `aws ec2 describe-instances` or EC2 metadata, updating `.env` and client scripts automatically.

### 7.3 Step-by-Step Implementation Sequence
1. Check AWS credentials and allocate EIP if available.
2. If EIP is allocated, bind it to instance `i-0720d9efd39dc72a8`.
3. Update `scripts/demo_lifecycle.py` and documentation with the permanent IP.

---

## 8. Prioritized Implementation Sequence & Execution Order

We recommend executing the remediation in this specific order to maximize immediate stability before introducing algorithmic enhancements:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 EXECUTION SEQUENCE                                     │
├─────────┬──────────────────────────────────────────────────────────┬───────────────────┤
│ Step 1  │ Initiative 5: Systemd Service Deployment on EC2          │ Operational Base  │
│ Step 2  │ Initiative 3: Durable SQLite Persistence for Zep Memory  │ Data Integrity    │
│ Step 3  │ Initiative 1: RouteLLM ONNX Preference Head Integration  │ Router Precision  │
│ Step 4  │ Initiative 2: Multi-Turn Context-Aware Semantic Vector   │ Cache Quality     │
│ Step 5  │ Initiative 4: Local SLM Upgrade to llama3.2:3b in Ollama │ Cognitive Score   │
│ Step 6  │ Initiative 6: Permanent Static IP / Endpoint Resolution  │ Networking        │
└─────────┴──────────────────────────────────────────────────────────┴───────────────────┘
```
