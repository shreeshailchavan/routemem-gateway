# AWS Cloud Development & Production Deployment Guide for RouteMem AI Gateway

This comprehensive guide details how to provision, configure, connect to, and run the **RouteMem AI Gateway** using Amazon Web Services (AWS) infrastructure for both cost-optimized development and high-concurrency production deployments.

---

## 1. AWS Architecture Overview

The RouteMem deployment topology on AWS decouples the lightweight **Gateway Control Plane** from the **GPU Inference Worker Cluster**, allowing independent scaling and cost optimization:

```
[ Local Machine / VS Code ]
           │
           │ (Secure AWS SSM Tunnel / SSH - No Open Port 22)
           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ AWS VPC (Virtual Private Cloud)                                                        │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ CONTROL PLANE NODE: AWS Graviton3 (t4g.xlarge — 4 vCPUs, 16GB RAM)               │  │
│  │ • FastAPI Ingestion & OpenAI/Anthropic API Proxy (Port 8000)                      │  │
│  │ • Tier-0 Redis SHA-256 Exact Hash Cache (Port 6379)                               │  │
│  │ • Tier-1 Qdrant HNSW Semantic Vector DB (Port 6333)                               │  │
│  │ • Tier-2 Zep Graphiti Temporal Knowledge Graph Memory (Port 8080)                 │  │
│  │ • DeBERTa-v3 ONNX Profiler & OmniRouter Lagrangian Dual Solver                   │  │
│  └────────────────────────────────────────┬─────────────────────────────────────────┘  │
│                                           │ Internal VPC Traffic                       │
│                                           ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ INFERENCE WORKER CLUSTER: NVIDIA GPU Node (g5.xlarge / g6.xlarge — Spot/On-Demand)│  │
│  │ • vLLM / SGLang Engine with RadixAttention KV Cache Reuse                         │  │
│  │ • Llama-3.1-8B-Instruct & Qwen-2.5-Coder-32B Models                              │  │
│  │ • LMCache Host System RAM KV Offloading                                          │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Infrastructure Sizing & Monthly Cost Models

### A. Development / Prototyping Environment (Cost-Optimized)
* **Strategy**: Control plane runs on low-cost Graviton3 ARM instances. GPU worker nodes run as **AWS Spot Instances** launched on-demand only during testing cycles.
* **Estimated AWS Monthly Bill**: **$75.00 – $120.00 USD / month**

| Component | AWS Resource | Pricing Model | Dev Usage Pattern | Monthly Spend |
| :--- | :--- | :--- | :--- | :--- |
| **Control Plane** | `t4g.xlarge` (4 vCPUs, 16GB RAM) | On-Demand ($0.134/hr) | Active ~8 hrs/day (176 hrs/mo) | $23.60 |
| **EBS Storage** | 100GB GP3 NVMe SSD | Persistent ($0.08/GB-mo) | 24/7 Vector DB & Cache Index | $8.00 |
| **Router Profiling** | `g6.xlarge` (1x NVIDIA L4 24GB) | Spot ($0.30/hr) | Active ~30 hrs/month | $9.00 |
| **SLM Test Engine** | `g5.xlarge` (1x NVIDIA A10G 24GB) | Spot ($0.40/hr) | Active ~50 hrs/month | $20.00 – $50.00 |
| **Cloud Fallbacks** | OpenAI / Anthropic APIs | Pay-as-you-go | ~5,000 test queries | $15.00 – $30.00 |

### B. Dedicated Production Environment (24/7 High-Availability)
* **Strategy**: 24/7 dedicated control plane and auto-scaling GPU inference workers with fallback to external cloud APIs.
* **Estimated AWS Monthly Bill**: **$525.00 – $725.00 USD / month**

---

## 3. Step-by-Step AWS Setup & Provisioning

### Step 1: Launch the Control Plane Instance
1. Open the **AWS EC2 Console** and select your preferred region (e.g., `us-east-1`).
2. Click **Launch Instance** and configure:
   * **Name**: `routemem-control-plane`
   * **AMI**: Ubuntu Server 22.04 LTS (ARM64 for Graviton3 or x86_64)
   * **Instance Type**: `t4g.xlarge` (4 vCPUs, 16GB RAM)
   * **Storage**: 100 GiB GP3 SSD (3,000 IOPS, 125 MB/s throughput)
   * **IAM Role**: Attach an IAM role with `AmazonSSMManagedInstanceCore` policy enabled.

### Step 2: Launch the GPU Inference Worker Instance
1. Click **Launch Instance** and configure:
   * **Name**: `routemem-gpu-worker`
   * **AMI**: Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04)
   * **Instance Type**: `g5.xlarge` (1x NVIDIA A10G 24GB VRAM) or `g6.xlarge` (1x NVIDIA L4 24GB VRAM)
   * **Purchasing Option**: Check **Spot Instance** for development (~60% discount).
   * **Storage**: 150 GiB GP3 SSD

### Step 3: Configure Security Groups
Create an internal Security Group (`sg-routemem-internal`) allowing intra-VPC communication:
* Allow **Inbound TCP 8000** (Gateway API) from VPC CIDR (`172.31.0.0/16`).
* Allow **Inbound TCP 6379** (Redis) and **TCP 6333** (Qdrant) from VPC CIDR.
* Allow **Inbound TCP 8001** (vLLM Inference Engine) from Control Plane IP.

---

## 4. Secure Connection from Local Machine (VS Code)

To develop locally while accessing AWS resources securely without exposing port 22 to the public internet, use **AWS Systems Manager (SSM)** or **VS Code Remote-SSH**.

### Option A: VS Code Remote-SSH (Recommended Developer Experience)
1. Install the **Remote - SSH** extension in VS Code on your local machine.
2. Edit your local SSH config file (`~/.ssh/config`):
   ```text
   Host routemem-aws
       HostName <AWS_EC2_PUBLIC_IP>
       User ubuntu
       IdentityFile ~/.ssh/routemem-key.pem
       ServerAliveInterval 60
   ```
3. Connect in VS Code (`Cmd+Shift+P` -> `Remote-SSH: Connect to Host...` -> `routemem-aws`). Your local VS Code now edits and runs code directly on the AWS instance.

### Option B: Local Port Forwarding via AWS SSM (Zero Public Ports)
Run SSH/SSM port forwarding from your local terminal to access AWS databases and APIs on `localhost`:
```bash
# Install AWS Session Manager Plugin locally
brew install --cask session-manager-plugin

# Tunnel AWS Gateway (8000) and Qdrant Dashboard (6333) to local machine
aws ssm start-session \
  --target i-0123456789abcdef0 \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters '{"host":["localhost"],"portNumber":["6333"],"localPortNumber":["6333"]}'
```
Now opening `http://localhost:6333/dashboard` in your local browser views the live Qdrant vector database on AWS.

---

## 5. Deployment & Execution Instructions

### Step 1: Environment Configuration
Create a `.env` file on the AWS Control Plane instance:
```env
# Server Config
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development

# Database Connections
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333

# Inference Backends
LOCAL_SLM_URL=http://172.31.18.42:8001/v1  # Internal IP of GPU Worker
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...

# Gateway Parameters
EXACT_CACHE_ENABLED=true
SEMANTIC_CACHE_THRESHOLD=0.95
COMPRESSION_RATIO=0.20
MONTHLY_BUDGET_LIMIT_USD=500.00
```

### Step 2: Start Control Plane Services via Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: routemem-redis
    ports:
      - "6379:6379"
    restart: always

  qdrant:
    image: qdrant/qdrant:latest
    container_name: routemem-qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
    restart: always

  gateway:
    build: .
    container_name: routemem-gateway
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - redis
      - qdrant
    restart: always
```

Run containers in detached mode:
```bash
docker-compose up -d
```

### Step 3: Start vLLM Local SLM Worker on GPU Instance
On the GPU instance (`g5.xlarge`), launch the high-throughput vLLM engine:
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8001 \
  --enable-prefix-caching \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

---

## 6. Verification & Health Monitoring

Verify end-to-end routing execution from your local machine:

```bash
# Test OpenAI-compatible chat completion endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "routemem-auto",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

### Expected Response
```json
{
  "id": "cmpl-routemem-9f28a",
  "object": "chat.completion",
  "created": 1728000000,
  "model": "Llama-3.1-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 7,
    "total_tokens": 19
  },
  "routemem_metadata": {
    "cache_status": "EXACT_MISS_SLM_HIT",
    "ttft_ms": 22.4,
    "cost_usd": 0.000002
  }
}
```

---

## 7. Cost Control & Best Practices Summary

1. **Auto-Stop GPU Instances**: Set up AWS CloudWatch alarms to shut down Spot `g5.xlarge` GPU instances if CPU/GPU utilization drops below 5% for 30 minutes.
2. **Graviton3 ARM Architecture**: Always use `t4g.xlarge` instances for Redis, Qdrant, and FastAPI control planes to save ~20% compared to x86 instances.
3. **Budget Alerts**: Configure AWS Budgets to send Slack or email alerts if monthly spend exceeds **$100.00** during development.
