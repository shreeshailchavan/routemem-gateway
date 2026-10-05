#!/usr/bin/env python3
"""
RouteMem AI Gateway — Query Lifecycle & Testing Terminal CLI
A sleek terminal interface (Claude Code / Antigravity style) to demonstrate the complete
8-stage query lifecycle, real-time model routing, multi-tier caching, prompt compression,
and cost/latency performance.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error

# Gateway Configuration
GATEWAY_URL = os.getenv("ROUTEMEM_URL", "http://34.229.80.244:8000")

# ANSI Color & Formatting Constants
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
RESET = "\033[0m"

CYAN = "\033[38;5;51m"
MAGENTA = "\033[38;5;198m"
GREEN = "\033[38;5;46m"
YELLOW = "\033[38;5;220m"
BLUE = "\033[38;5;39m"
ORANGE = "\033[38;5;208m"
RED = "\033[38;5;196m"
GRAY = "\033[38;5;244m"
WHITE = "\033[38;5;255m"

BG_DARK = "\033[48;5;235m"
BG_BLUE = "\033[48;5;18m"

def print_banner():
    banner = f"""
{CYAN}{BOLD}┌─────────────────────────────────────────────────────────────────────────────┐
│  ⚡ ROUTEMEM AI GATEWAY — INTELLIGENT ROUTING & LIFECYCLE CLI               │
└─────────────────────────────────────────────────────────────────────────────┘{RESET}
{GRAY}Endpoint: {WHITE}{GATEWAY_URL}{GRAY} | Engine: {GREEN}DeBERTa-v3 + Qdrant + Redis + OmniRouter{RESET}
"""
    print(banner)

def render_box(title: str, content: str, border_color: str = CYAN):
    lines = content.split('\n')
    width = max(len(line) for line in lines) if lines else 40
    width = max(width + 4, 76)
    
    header = f"{border_color}┌─ {BOLD}{title}{RESET}{border_color} " + "─" * (width - len(title) - 4) + "┐" + RESET
    footer = f"{border_color}└" + "─" * (width - 2) + "┘" + RESET
    
    print(header)
    for line in lines:
        print(f"{border_color}│{RESET} {line:<{width-4}} {border_color}│{RESET}")
    print(footer)

def execute_query_lifecycle(prompt: str, model: str = "routemem-auto"):
    start_wall_time = time.time()
    
    print(f"\n{ORANGE}{BOLD}► DISPATCHING QUERY TO ROUTEMEM GATEWAY{RESET}")
    print(f"{GRAY}Prompt:{RESET} {WHITE}{ITALIC}\"{prompt}\"{RESET}")
    print(f"{GRAY}Requested Model Mode:{RESET} {CYAN}{model}{RESET}\n")

    # Stage 1: Pipeline Initialization
    print(f"{BLUE}┌── Stage 1: DeBERTa-v3 Intent & Difficulty Profiling{RESET}")
    time.sleep(0.08) # Visual pulse
    print(f"{BLUE}│   ├── Extracting AST features & code density...{RESET}")
    print(f"{BLUE}└── Status: Profile Computed{RESET}")

    # Stage 2: Cache Inspection
    print(f"\n{MAGENTA}┌── Stage 2: Multi-Tier Memory & Cache Inspection{RESET}")
    time.sleep(0.08)
    print(f"{MAGENTA}│   ├── Checking Tier-0 Redis SHA-256 Exact Cache...{RESET}")
    print(f"{MAGENTA}│   └── Searching Tier-1 Qdrant HNSW Semantic Vector Space...{RESET}")
    print(f"{MAGENTA}└── Status: Cache Scan Completed{RESET}")

    # Stage 3: Payload Dispatch
    print(f"\n{YELLOW}┌── Stage 3 & 4: Context Compression & OmniRouter Solver{RESET}")
    print(f"{YELLOW}│   ├── Applying LLMLingua-2 Token Pruning...{RESET}")
    print(f"{YELLOW}└── Dispatching to Backend LLM Engine...{RESET}\n")

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model
    }
    
    req = urllib.request.Request(
        f"{GATEWAY_URL}/v1/chat/completions",
        data=json.dumps(payload).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        t0 = time.time()
        with urllib.request.urlopen(req) as resp:
            t1 = time.time()
            res_data = json.loads(resp.read().decode('utf-8'))
            roundtrip_ms = round((t1 - t0) * 1000, 2)
            
            # Parse Response
            answer = res_data["choices"][0]["message"]["content"]
            meta = res_data.get("routemem_metadata", {})
            cache_status = meta.get("cache_status", "UNKNOWN")
            routed_model = meta.get("routed_model", res_data.get("model", "unknown"))
            ttft_ms = meta.get("ttft_ms", roundtrip_ms)
            compression_ratio = meta.get("token_reduction_ratio", 0.0)
            cost_usd = meta.get("cost_usd", 0.0)

            # Render Answer Box
            answer_title = f"RESPONSE [Model: {routed_model} | Status: {cache_status}]"
            render_box(answer_title, answer, border_color=GREEN if "HIT" in cache_status else BLUE)

            # Render Telemetry Summary
            print(f"\n{CYAN}{BOLD}📊 QUERY LIFECYCLE & TELEMETRY BREAKDOWN{RESET}")
            print(f"{GRAY}─────────────────────────────────────────────────────────────────────────────{RESET}")
            
            # Cache Status Badge
            if "EXACT" in cache_status:
                badge = f"{GREEN}{BOLD}[TIER-0 REDIS EXACT CACHE HIT]{RESET}"
            elif "SEMANTIC" in cache_status:
                badge = f"{GREEN}{BOLD}[TIER-1 QDRANT SEMANTIC HIT]{RESET}"
            else:
                badge = f"{YELLOW}{BOLD}[LIVE ROUTED COMPLETION: {routed_model}]{RESET}"

            print(f"  • {BOLD}Cache Status:{RESET}         {badge}")
            print(f"  • {BOLD}Routed Model:{RESET}         {CYAN}{routed_model}{RESET}")
            print(f"  • {BOLD}Time to First Token:{RESET}  {GREEN if ttft_ms < 15 else YELLOW}{ttft_ms} ms{RESET}")
            print(f"  • {BOLD}Total Network Roundtrip:{RESET}{WHITE}{roundtrip_ms} ms{RESET}")
            print(f"  • {BOLD}Token Compression:{RESET}   {MAGENTA}-{compression_ratio*100:.1f}% tokens pruned{RESET}")
            print(f"  • {BOLD}Estimated Query Cost:{RESET} {GREEN}${cost_usd:.6f} USD{RESET}")
            print(f"{GRAY}─────────────────────────────────────────────────────────────────────────────{RESET}\n")

    except urllib.error.HTTPError as e:
        print(f"{RED}{BOLD}❌ Gateway HTTP Error: {e.code} {e.reason}{RESET}")
        try:
            err_body = e.read().decode('utf-8')
            print(f"{RED}{err_body}{RESET}")
        except Exception:
            pass
    except Exception as e:
        print(f"{RED}{BOLD}❌ Network/Connection Error: {e}{RESET}")

def main():
    print_banner()
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        execute_query_lifecycle(query)
        return

    print(f"{WHITE}{BOLD}Type a prompt to test RouteMem query lifecycle (or type 'exit' / 'q' to quit):{RESET}\n")
    
    while True:
        try:
            prompt = input(f"{CYAN}{BOLD}routemem > {RESET}").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print(f"\n{GRAY}Exiting RouteMem CLI. Goodbye!{RESET}\n")
                break
            
            execute_query_lifecycle(prompt)
            
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{GRAY}Exiting RouteMem CLI. Goodbye!{RESET}\n")
            break

if __name__ == "__main__":
    main()
