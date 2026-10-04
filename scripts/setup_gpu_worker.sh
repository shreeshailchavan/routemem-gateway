#!/bin/bash
# RouteMem AI Gateway — AWS GPU Worker Launch Script (g5.xlarge / NVIDIA A10G)
set -e

MODEL_NAME=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
PORT=${2:-8001}

echo "=== Launching vLLM Engine on GPU Worker Node ==="
echo "Model: $MODEL_NAME"
echo "Port:  $PORT"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --port "$PORT" \
  --enable-prefix-caching \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
