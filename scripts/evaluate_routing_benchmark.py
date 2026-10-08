#!/usr/bin/env python3
"""
RouteMem AI Gateway — Empirical Routing Evaluation & Metrics Engine
====================================================================
Evaluates the 8-stage RouteMem routing pipeline across multi-tier complexity
benchmarks (Syntactic AST + Dense Semantic Embeddings + RouteLLM Preference Head).

Calculates:
  - Overall Routing Accuracy
  - Category-Level Accuracy
  - Precision, Recall, and F1-Scores
  - Confusion Matrix
  - Cost Savings vs All-to-Cloud Baseline (GPT-4o)
  - Latency / TTFT Profile by Tier
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
from collections import defaultdict
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.router.profiler import QueryProfiler
from app.router.omnirouter import OmniRouter
from app.cache.dense_embedder import DenseEmbedder

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks", "large_scale_routing_benchmark.jsonl")
OUTPUT_REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "routing_evaluation_metrics.json")


def evaluate_routing(dataset_path: str = DATASET_PATH) -> Dict[str, Any]:
    print("=" * 78)
    print("  ROUTEMEM EMPIRICAL ROUTING ACCURACY & EVALUATION METRICS")
    print("=" * 78)

    profiler = QueryProfiler()
    router = OmniRouter()
    embedder = DenseEmbedder()

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # Categories to evaluate for LLM/SLM Routing
    categories = ["systems_architecture", "code_generation", "math_reasoning", "simple_qa", "domain_expert"]
    eval_records = [r for r in records if r.get("category") in categories]

    # Also evaluate cache tiers
    cache_records = [r for r in records if "cache" in r.get("category", "")]

    print(f"Loaded {len(eval_records)} routing benchmark queries across {len(categories)} categories.")
    print(f"Loaded {len(cache_records)} caching verification queries.")
    print("-" * 78)

    y_true_models = []
    y_pred_models = []
    y_true_categories = []
    y_pred_intents = []
    latencies_ms = []

    category_results = defaultdict(lambda: {"total": 0, "correct": 0, "mismatches": []})
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    all_models = set()

    for idx, item in enumerate(eval_records, 1):
        cat = item["category"]
        expected_model = item["expected_model"]
        prompt = item["prompt"]

        all_models.add(expected_model)

        t0 = time.perf_counter()
        query_vector = embedder.embed(prompt)
        difficulty, intent = profiler.profile(prompt, embedding=query_vector)
        selected_model = router.select_model(difficulty=difficulty, intent=intent)
        eval_time_ms = (time.perf_counter() - t0) * 1000

        latencies_ms.append(eval_time_ms)
        all_models.add(selected_model)

        is_correct = (selected_model == expected_model)
        category_results[cat]["total"] += 1
        if is_correct:
            category_results[cat]["correct"] += 1
        else:
            category_results[cat]["mismatches"].append({
                "prompt": prompt,
                "expected": expected_model,
                "predicted": selected_model,
                "difficulty": difficulty,
                "intent": intent
            })

        y_true_models.append(expected_model)
        y_pred_models.append(selected_model)
        y_true_categories.append(cat)
        y_pred_intents.append(intent)
        confusion_matrix[expected_model][selected_model] += 1

    # Metrics calculation
    total_samples = len(eval_records)
    total_correct = sum(1 for yt, yp in zip(y_true_models, y_pred_models) if yt == yp)
    overall_accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0

    # Per-model Precision, Recall, F1
    model_metrics = {}
    for model in sorted(all_models):
        tp = confusion_matrix[model][model]
        fp = sum(confusion_matrix[other][model] for other in all_models if other != model)
        fn = sum(confusion_matrix[model][other] for other in all_models if other != model)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        model_metrics[model] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": tp + fn
        }

    # Macro & Weighted F1
    macro_f1 = float(np.mean([m["f1_score"] for m in model_metrics.values()]))
    weighted_f1 = float(sum(m["f1_score"] * m["support"] for m in model_metrics.values()) / total_samples) if total_samples > 0 else 0.0

    # Cost Analysis ($0.03 baseline GPT-4o vs RouteMem local SLM $0.00 + Cloud Frontier $0.003)
    cloud_model_count = sum(1 for m in y_pred_models if "claude" in m or "gpt" in m or "gemini" in m)
    local_model_count = total_samples - cloud_model_count
    baseline_cost = total_samples * 0.030000  # All routed to GPT-4o
    routemem_cost = (cloud_model_count * 0.003000) + (local_model_count * 0.000000)
    cost_savings_pct = ((baseline_cost - routemem_cost) / baseline_cost * 100.0) if baseline_cost > 0 else 0.0

    # Display Results
    print(f"\n{BOLD_S}OVERALL ROUTING ACCURACY:{RESET_S} {total_correct}/{total_samples} ({overall_accuracy * 100:.2f}%)")
    print(f"{BOLD_S}MACRO F1 SCORE:{RESET_S}          {macro_f1:.4f}")
    print(f"{BOLD_S}WEIGHTED F1 SCORE:{RESET_S}       {weighted_f1:.4f}")
    print(f"{BOLD_S}MEAN ROUTER OVERHEAD:{RESET_S}    {np.mean(latencies_ms):.3f} ms (p95: {np.percentile(latencies_ms, 95):.3f} ms)")
    print(f"{BOLD_S}COST REDUCTION VS GPT-4o:{RESET_S} {cost_savings_pct:.1f}% (${routemem_cost:.4f} vs ${baseline_cost:.4f})")
    print("\n" + "=" * 78)
    print(f"{'CATEGORY':<25} {'SAMPLES':<10} {'CORRECT':<10} {'ACCURACY':<12}")
    print("-" * 78)
    cat_summary = {}
    for cat in categories:
        st = category_results[cat]
        acc = (st["correct"] / st["total"] * 100) if st["total"] > 0 else 0.0
        cat_summary[cat] = {"total": st["total"], "correct": st["correct"], "accuracy": acc}
        print(f"{cat:<25} {st['total']:<10} {st['correct']:<10} {acc:>6.1f}%")

    print("\n" + "=" * 78)
    print(f"{'TARGET MODEL':<24} {'PRECISION':<12} {'RECALL':<12} {'F1 SCORE':<12} {'SUPPORT':<10}")
    print("-" * 78)
    for model, m in model_metrics.items():
        print(f"{model:<24} {m['precision']:>9.4f}   {m['recall']:>9.4f}   {m['f1_score']:>9.4f}   {m['support']:>8}")

    print("\n" + "=" * 78)
    print("CONFUSION MATRIX (Rows: Ground Truth Expected -> Columns: RouteMem Routed)")
    print("-" * 78)
    header = f"{'Expected Model':<24} | " + " | ".join(f"{m[:14]:<14}" for m in sorted(all_models))
    print(header)
    print("-" * len(header))
    for true_m in sorted(all_models):
        row_str = f"{true_m:<24} | "
        counts = [f"{confusion_matrix[true_m][pred_m]:<14}" for pred_m in sorted(all_models)]
        print(row_str + " | ".join(counts))

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": os.path.basename(dataset_path),
        "total_queries": total_samples,
        "overall_accuracy": round(overall_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "cost_analysis": {
            "baseline_gpt4o_cost_usd": round(baseline_cost, 4),
            "routemem_cost_usd": round(routemem_cost, 4),
            "cost_savings_pct": round(cost_savings_pct, 2),
            "local_slm_queries": local_model_count,
            "cloud_frontier_queries": cloud_model_count
        },
        "category_accuracy": cat_summary,
        "model_metrics": model_metrics,
        "confusion_matrix": {k: dict(v) for k, v in confusion_matrix.items()}
    }

    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n✔ Full evaluation metrics serialized to: {OUTPUT_REPORT_PATH}")
    return report

BOLD_S = "\033[1m"
RESET_S = "\033[0m"

if __name__ == "__main__":
    evaluate_routing()
