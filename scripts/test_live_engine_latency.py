#!/usr/bin/env python3
"""
RouteMem Live Engine Latency & TTFT Tester
Measures cold SLM generation, cloud inference, exact cache, and semantic cache
directly on localhost:8000 without cross-continental network latency.
"""

import sys
import os
import time
import json
import urllib.request

GATEWAY_URL = os.getenv("ROUTEMEM_URL", "http://localhost:8000")

def run():
    print("=========================================================================================")
    print("⚡ ROUTEMEM PURE HARDWARE & ENGINE LATENCY BENCHMARK (ON EC2)")
    print(f"Target Endpoint: {GATEWAY_URL}")
    print("=========================================================================================\n")

    # 1. Clear all cache tiers
    print("[*] Flushing Redis, Qdrant, and Zep Memory to guarantee clean cold measurement...")
    try:
        with urllib.request.urlopen(f"{GATEWAY_URL}/v1/admin/cache/clear") as resp:
            print("    ✔ Caches successfully cleared.\n")
    except Exception as e:
        print(f"    [!] Cache clear notice: {e}\n")

    tests = [
        ("Tier-3 routemem-specialist (Cold SLM)", "Explain how Linux epoll edge-triggered mode prevents busy-waiting loops.", "routemem-specialist"),
        ("Tier-3 qwen2.5-coder:3b (Cold Code SLM)", "Write a Python generator function to stream prime numbers up to n using a sieve.", "qwen2.5-coder:3b"),
        ("Tier-3 deepseek-r1:1.5b (Cold Reasoning SLM)", "There are 5 birds on a wire. A hunter shoots 1. How many birds are left? Explain step by step.", "deepseek-r1:1.5b"),
        ("Tier-A Groq LPU / Cloud API (Cold)", "Explain the legal standards under the Lanham Act for trademark dilution versus consumer confusion.", "openai/gpt-oss-120b"),
        ("Tier-0 Redis Exact Cache (Repeat Query)", "Explain how Linux epoll edge-triggered mode prevents busy-waiting loops.", "routemem-auto"),
        ("Tier-1 Qdrant Semantic Cache (Paraphrase)", "How does edge-triggered epoll in the Linux kernel avoid busy waiting?", "routemem-auto")
    ]

    print("=" * 105)
    print(f"{'Tier / Model Under Test':<44} | {'Cache Status':<16} | {'TTFT (ms)':<10} | {'Throughput':<12} | {'Total Duration'}")
    print("=" * 105)

    results = []

    for label, prompt, model in tests:
        if "Cache" in label:
            # Brief pause to allow Stage 8 async background sync to finish indexing
            time.sleep(1.2)

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": model,
            "stream": True,
            "max_tokens": 35
        }
        req = urllib.request.Request(
            f"{GATEWAY_URL}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        t0 = time.perf_counter()
        t_first = None
        tokens = 0
        cache_status = "LOCAL_SLM_HIT" if "Tier-3" in label else "EXACT_MISS"

        with urllib.request.urlopen(req, timeout=35) as resp:
            ct = resp.headers.get("content-type", "")
            if "application/json" in ct:
                data = json.loads(resp.read().decode("utf-8"))
                t_first = time.perf_counter()
                meta = data.get("routemem_metadata", {})
                cache_status = meta.get("cache_status", "EXACT_HIT")
                server_ttft = meta.get("ttft_ms", (t_first - t0) * 1000.0)
                tokens = len(data.get("choices", [{}])[0].get("message", {}).get("content", "").split())
                t1 = time.perf_counter()
                ttft = server_ttft
                tps = 0.0
            else:
                for line in resp:
                    l = line.decode("utf-8").strip()
                    if not l.startswith("data: ") or l == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(l[6:])
                        meta = chunk.get("routemem_metadata", {})
                        if meta:
                            cache_status = meta.get("cache_status", cache_status)
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            if t_first is None:
                                t_first = time.perf_counter()
                            tokens += 1
                    except json.JSONDecodeError:
                        pass
                t1 = time.perf_counter()
                ttft = ((t_first - t0) * 1000.0) if t_first else ((t1 - t0) * 1000.0)
                decode_s = (t1 - t_first) if t_first else 0.0
                tps = (tokens / decode_s) if decode_s > 0 else 0.0

        dur = (t1 - t0) * 1000.0
        print(f"{label:<44} | {cache_status:<16} | {ttft:8.2f} ms | {tps:7.1f} tok/s | {dur:8.2f} ms")
        results.append({
            "tier": label,
            "cache_status": cache_status,
            "ttft_ms": round(ttft, 2),
            "tokens_per_sec": round(tps, 1),
            "total_duration_ms": round(dur, 2)
        })

    print("=" * 105)

    # Save to JSON
    out_file = "reports/live_engine_latency_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "results": results}, f, indent=2)
    print(f"\n✔ Results saved to: {out_file}\n")

if __name__ == "__main__":
    run()
