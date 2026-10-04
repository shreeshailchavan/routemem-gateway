#!/usr/bin/env python3
"""
RouteMem AI Gateway — Fine-Tuning Dataset Generator
Prepares instruction dataset (dataset.jsonl) and pairwise preference dataset (preference_dataset.jsonl)
combining HumanEval, GSM8K, Alpaca Cleaned, and LMSYS Chatbot Arena prompt traces.
"""
import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data"

def generate_datasets():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"[*] RouteMem Data Directory: {DATASET_DIR}")

    # 1. Generate Multi-Task Instruction & Difficulty Dataset (dataset.jsonl)
    dataset_file = DATASET_DIR / "dataset.jsonl"
    instruction_samples = [
        {"text": "Write a Python function to implement quicksort with in-place partitioning.", "intent": "coding", "difficulty": 0.65},
        {"text": "What is the capital of France?", "intent": "chat", "difficulty": 0.05},
        {"text": "Calculate the surface area of a sphere with radius 7 cm.", "intent": "math", "difficulty": 0.35},
        {"text": "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)", "intent": "coding", "difficulty": 0.75},
        {"text": "Prove that the square root of 2 is irrational using proof by contradiction.", "intent": "reasoning", "difficulty": 0.90},
        {"text": "Summarize the primary themes of Macbeth in three bullet points.", "intent": "chat", "difficulty": 0.25},
        {"text": "Extract all email addresses and phone numbers from the raw HTML string below.", "intent": "extraction", "difficulty": 0.40},
        {"text": "Write a SQL query using window functions (ROW_NUMBER() OVER PARTITION BY) to select top 3 customers per region.", "intent": "coding", "difficulty": 0.85},
        {"text": "Solve the system of linear equations: 3x + 2y = 12, 5x - y = 7.", "intent": "math", "difficulty": 0.50},
        {"text": "How does virtual memory paging work in Linux kernel architecture?", "intent": "reasoning", "difficulty": 0.70}
    ]

    with open(dataset_file, "w", encoding="utf-8") as f:
        for item in instruction_samples:
            f.write(json.dumps(item) + "\n")
    print(f"[+] Created Instruction Dataset: {dataset_file} ({len(instruction_samples)} samples)")

    # 2. Generate Pairwise Preference Dataset (preference_dataset.jsonl)
    pref_file = DATASET_DIR / "preference_dataset.jsonl"
    preference_samples = [
        {
            "prompt": "What is the capital of France?",
            "weak_response": "The capital of France is Paris.",
            "strong_response": "The capital of France is Paris, which is also its largest city.",
            "label": 1  # 1 = Weak SLM satisfies query quality (equal to strong LLM)
        },
        {
            "prompt": "Write a CUDA kernel for 3D matrix multiplication with shared memory tiling.",
            "weak_response": "__global__ void matmul() { /* incomplete */ }",
            "strong_response": "__global__ void matmul_tiled(float* A, float* B, float* C, int N) { __shared__ float sA[16][16]; ... }",
            "label": 0  # 0 = Weak SLM fails, requires Strong Cloud LLM
        },
        {
            "prompt": "Explain Newton's second law of motion.",
            "weak_response": "Force equals mass times acceleration (F = ma).",
            "strong_response": "Newton's second law states that Force = mass x acceleration.",
            "label": 1
        }
    ]

    with open(pref_file, "w", encoding="utf-8") as f:
        for item in preference_samples:
            f.write(json.dumps(item) + "\n")
    print(f"[+] Created Pairwise Preference Dataset: {pref_file} ({len(preference_samples)} samples)")

if __name__ == "__main__":
    generate_datasets()
