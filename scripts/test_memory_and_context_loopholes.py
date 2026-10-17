#!/usr/bin/env python3
"""
RouteMem AI Gateway — Brutal Context & Zep Graphiti Memory Penetration Test
=============================================================================
Stress-tests:
  1. Multi-turn zero-shot factual recall via Zep Graphiti & SQLite Memory
  2. OpenAI messages[] history propagation to backend SLMs
  3. Cross-session memory leakage & isolation
  4. Cache collision on identical prompts across different sessions
  5. Asynchronous graph ingestion latency & fallback recall
"""

import sys
import time
import json
import uuid
import urllib.request
import urllib.error

GATEWAY_URL = "http://54.221.136.83:8000"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def post_chat(messages, session_id=None, stream=False, user=None):
    payload = {
        "messages": messages,
        "model": "routemem-auto",
        "stream": stream
    }
    if session_id:
        payload["session_id"] = session_id
    if user:
        payload["user"] = user

    req = urllib.request.Request(
        f"{GATEWAY_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            latency_ms = (time.time() - t0) * 1000
            content = data["choices"][0]["message"]["content"]
            meta = data.get("routemem_metadata", {})
            return content, meta, latency_ms, None
    except Exception as e:
        return None, {}, (time.time() - t0) * 1000, str(e)


def run_brutal_tests():
    print(f"\n{BOLD}{CYAN}{'='*80}{RESET}")
    print(f"{BOLD}{CYAN}  ROUTEMEM BRUTAL CONTEXT & ZEP GRAPHITI MEMORY PENETRATION TEST{RESET}")
    print(f"{BOLD}{CYAN}{'='*80}{RESET}\n")

    # Flush caches first for clean test
    clear_req = urllib.request.Request(f"{GATEWAY_URL}/v1/admin/cache/clear", data=b"{}", headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(clear_req, timeout=5)
        print(f"{GREEN}✔ Memory & Caches flushed on gateway.{RESET}\n")
    except Exception as e:
        print(f"{RED}✖ Failed to clear gateway cache: {e}{RESET}\n")

    test_session = f"brutal-test-{uuid.uuid4().hex[:8]}"
    print(f"{BOLD}Target Session ID:{RESET} {YELLOW}{test_session}{RESET}")

    # =========================================================================
    # TEST 1: Temporal Fact Injection & Immediate Recall
    # =========================================================================
    print(f"\n{BOLD}{CYAN}[TEST 1] Fact Injection & Zero-Shot Recall in Same Session{RESET}")
    fact_turn_1 = "Remember this private detail: Our internal code-name is Project-Valkyrie-99 and our team lead is Dr. Elena Rostova."
    print(f"  Step 1. Injecting Fact into Session '{test_session}'...")
    content1, meta1, lat1, err1 = post_chat([{"role": "user", "content": fact_turn_1}], session_id=test_session)
    if err1:
        print(f"  {RED}✖ Ingestion failed: {err1}{RESET}")
    else:
        print(f"  Response: {content1[:100]}... ({lat1:.1f}ms)")
        print(f"  Routed Model: {meta1.get('actual_answering_model')} | Status: {meta1.get('cache_status')}")

    # Wait 2 seconds for async background sync
    time.sleep(2.0)

    # Turn 2: Probe the fact
    query_turn_2 = "What is our internal code-name and who is our team lead?"
    print(f"\n  Step 2. Probing Fact in Session '{test_session}'...")
    content2, meta2, lat2, err2 = post_chat([{"role": "user", "content": query_turn_2}], session_id=test_session)
    if err2:
        print(f"  {RED}✖ Query failed: {err2}{RESET}")
    else:
        print(f"  Response: {BOLD}{content2.strip()}{RESET}")
        print(f"  KG Facts Retrieved: {meta2.get('kg_facts_retrieved')} | KG Used: {meta2.get('kg_memory_used')}")
        passed_valkyrie = "valkyrie" in content2.lower()
        passed_elena = "elena" in content2.lower() or "rostova" in content2.lower()
        if passed_valkyrie and passed_elena:
            print(f"  {GREEN}{BOLD}✔ TEST 1 PASSED: Fact successfully retrieved from memory and answered correctly!{RESET}")
        else:
            print(f"  {RED}{BOLD}✖ TEST 1 FAILED: Model could not recall 'Project-Valkyrie-99' or 'Dr. Elena Rostova'!{RESET}")

    # =========================================================================
    # TEST 2: Multi-Turn Conversation History (messages[] array)
    # =========================================================================
    print(f"\n{BOLD}{CYAN}[TEST 2] Multi-Turn Message Array Context Awareness{RESET}")
    test_session_2 = f"brutal-conv-{uuid.uuid4().hex[:8]}"
    messages_history = [
        {"role": "user", "content": "I am thinking of a secret number. It is 42."},
        {"role": "assistant", "content": "Understood! I will remember that your secret number is 42."},
        {"role": "user", "content": "What was the number I just mentioned?"}
    ]
    print(f"  Sending multi-turn messages array (3 messages) to session '{test_session_2}'...")
    content_conv, meta_conv, lat_conv, err_conv = post_chat(messages_history, session_id=test_session_2)
    if err_conv:
        print(f"  {RED}✖ Query failed: {err_conv}{RESET}")
    else:
        print(f"  Response: {BOLD}{content_conv.strip()}{RESET}")
        if "42" in content_conv:
            print(f"  {GREEN}{BOLD}✔ TEST 2 PASSED: Model maintained in-context multi-turn awareness!{RESET}")
        else:
            print(f"  {RED}{BOLD}✖ TEST 2 FAILED: Model did not receive or recall conversation history (missed '42')!{RESET}")

    # =========================================================================
    # TEST 3: Cross-Session Isolation (Data Leakage Test)
    # =========================================================================
    print(f"\n{BOLD}{CYAN}[TEST 3] Cross-Session Security & Leakage Prevention{RESET}")
    unrelated_session = f"stranger-{uuid.uuid4().hex[:8]}"
    print(f"  Probing private facts from an UNRELATED session '{unrelated_session}'...")
    content_leak, meta_leak, lat_leak, err_leak = post_chat([{"role": "user", "content": "What is our internal code-name and who is our team lead?"}], session_id=unrelated_session)
    if err_leak:
        print(f"  {RED}✖ Query failed: {err_leak}{RESET}")
    else:
        print(f"  Response: {BOLD}{content_leak.strip()}{RESET}")
        print(f"  KG Facts Retrieved: {meta_leak.get('kg_facts_retrieved')}")
        leaked = "valkyrie" in content_leak.lower() or "rostova" in content_leak.lower()
        if leaked:
            print(f"  {RED}{BOLD}✖ TEST 3 CRITICAL SECURITY VULNERABILITY: Facts leaked across sessions!{RESET}")
        else:
            print(f"  {GREEN}{BOLD}✔ TEST 3 PASSED: Clean session isolation. No cross-tenant fact leakage.{RESET}")

    # =========================================================================
    # TEST 4: Semantic Cache Collision Across Different Sessions
    # =========================================================================
    print(f"\n{BOLD}{CYAN}[TEST 4] Semantic Cache False-Hit on Context-Dependent Prompts{RESET}")
    sess_a = f"sess-a-{uuid.uuid4().hex[:6]}"
    sess_b = f"sess-b-{uuid.uuid4().hex[:6]}"

    print(f"  Session A: Setting context 'My name is Alice'...")
    post_chat([{"role": "user", "content": "My name is Alice."}], session_id=sess_a)
    time.sleep(1.0)
    ans_a, _, _, _ = post_chat([{"role": "user", "content": "What is my name?"}], session_id=sess_a)
    print(f"  Session A Answer: {ans_a.strip()[:60]}")

    print(f"  Session B: Setting context 'My name is Bob'...")
    post_chat([{"role": "user", "content": "My name is Bob."}], session_id=sess_b)
    time.sleep(1.0)
    ans_b, meta_b, _, _ = post_chat([{"role": "user", "content": "What is my name?"}], session_id=sess_b)
    print(f"  Session B Answer: {ans_b.strip()[:60]}")
    print(f"  Session B Cache Status: {meta_b.get('cache_status')}")

    if "alice" in ans_b.lower():
        print(f"  {RED}{BOLD}✖ TEST 4 FAILED: Semantic cache returned Alice's answer to Bob! Cache poisoning!{RESET}")
    elif "bob" in ans_b.lower():
        print(f"  {GREEN}{BOLD}✔ TEST 4 PASSED: Context correctly resolved to Bob! No cross-session cache poisoning.{RESET}")
    else:
        print(f"  {YELLOW}⚠ TEST 4 INCONCLUSIVE: Answer did not contain Bob or Alice ({ans_b.strip()[:60]}).{RESET}")


if __name__ == "__main__":
    run_brutal_tests()
