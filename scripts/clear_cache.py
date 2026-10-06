#!/usr/bin/env python3
"""
RouteMem AI Gateway — Cache & Memory Flushing Utility.

Flushes all memory tiers:
1. Tier-0 Redis SHA-256 Exact Hash Cache
2. Tier-1 Qdrant HNSW Semantic Vector Collection
3. Tier-2 Zep Graphiti Temporal Knowledge Graph Memory
"""

import os
import sys
import json
import urllib.request
import urllib.error

GATEWAY_URL = os.getenv("ROUTEMEM_URL", "http://18.214.99.62:8000")

GREEN = "\033[38;5;82m"
CYAN = "\033[38;5;51m"
RED = "\033[38;5;196m"
YELLOW = "\033[38;5;220m"
BOLD = "\033[1m"
RESET = "\033[0m"

def clear_gateway_caches(gateway_url: str = GATEWAY_URL):
    print(f"\n{CYAN}{BOLD}⚡ ROUTEMEM MEMORY PURGE UTILITY{RESET}")
    print(f"Target Gateway: {YELLOW}{gateway_url}{RESET}\n")

    url = f"{gateway_url.rstrip('/')}/v1/admin/cache/clear"
    req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"{GREEN}{BOLD}✔ {data.get('message', 'Caches cleared successfully.')}{RESET}")
            details = data.get("details", {})
            for tier, status in details.items():
                status_color = GREEN if status == "cleared" else YELLOW
                print(f"  • {tier:<38}: {status_color}{BOLD}[{status.upper()}]{RESET}")
            print(f"\n{GREEN}{BOLD}All memory tiers are pristine and ready for fresh query benchmarks.{RESET}\n")
    except urllib.error.HTTPError as e:
        print(f"{RED}{BOLD}❌ HTTP Error {e.code}: {e.reason}{RESET}")
    except Exception as e:
        print(f"{RED}{BOLD}❌ Connection Error: {e}{RESET}")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else GATEWAY_URL
    clear_gateway_caches(url)
