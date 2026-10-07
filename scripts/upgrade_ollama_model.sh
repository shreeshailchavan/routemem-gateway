#!/usr/bin/env bash
# RouteMem AI Gateway — Local SLM Upgrade Script
# Upgrades native Ollama model from llama3.2:1b to llama3.2:3b on ARM CPU

set -e

TARGET_MODEL="llama3.2:3b"

echo "=== Upgrading Ollama Local SLM to $TARGET_MODEL ==="

# Check Ollama service
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "[!] Ollama service is not running. Starting ollama service..."
    sudo systemctl restart ollama.service
    sleep 3
fi

echo "[*] Pulling $TARGET_MODEL into Ollama..."
ollama pull "$TARGET_MODEL"

echo "[+] Successfully pulled $TARGET_MODEL!"
echo "[*] Available Ollama models:"
ollama list

echo "[*] Verifying test inference with $TARGET_MODEL..."
curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"$TARGET_MODEL\",
  \"prompt\": \"Explain quantum entanglement in 1 sentence.\",
  \"stream\": false
}" | jq -r '.response'

echo "[+] Ollama upgrade completed successfully!"
