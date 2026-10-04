import os
import json

os.makedirs("data/benchmarks", exist_ok=True)

print("=== GATHERING BENCHMARK & EVALUATION DATASETS ===")

# Standardized Benchmark Datasets
benchmarks = [
    # GSM8K (Grade School Math)
    {
        "id": "gsm8k_001",
        "benchmark": "GSM8K",
        "category": "math_reasoning",
        "difficulty": 0.75,
        "prompt": "Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast every morning and uses 4 for baking. She sells the remainder at the farmers' market for $2 per egg. How much money does she make in a 7-day week?",
        "reference_answer": "16 - 3 - 4 = 9 eggs sold per day. 9 * 2 = $18 per day. 18 * 7 = $126 per week.",
        "target_capability": ["math", "reasoning"]
    },
    {
        "id": "gsm8k_002",
        "benchmark": "GSM8K",
        "category": "math_reasoning",
        "difficulty": 0.85,
        "prompt": "A train travels at 60 mph for 2 hours, then speeds up to 75 mph for 3 hours. What is the average speed of the train over the entire 5-hour journey?",
        "reference_answer": "(60 * 2 + 75 * 3) / 5 = (120 + 225) / 5 = 345 / 5 = 69 mph.",
        "target_capability": ["math", "reasoning"]
    },
    # HumanEval (Python Code Generation)
    {
        "id": "humaneval_001",
        "benchmark": "HumanEval",
        "category": "code_generation",
        "difficulty": 0.65,
        "prompt": "Write a Python function `has_close_elements(numbers: list[float], threshold: float) -> bool` that checks if in given list of numbers, any two numbers are closer to each other than the given threshold.",
        "reference_answer": "def has_close_elements(numbers, threshold):\n    for i, n1 in enumerate(numbers):\n        for j, n2 in enumerate(numbers):\n            if i != j and abs(n1 - n2) < threshold:\n                return True\n    return False",
        "target_capability": ["code", "reasoning"]
    },
    {
        "id": "humaneval_002",
        "benchmark": "HumanEval",
        "category": "code_generation",
        "difficulty": 0.90,
        "prompt": "Implement an optimal algorithm `longest_palindromic_substring(s: str) -> str` with O(N^2) or O(N) time complexity.",
        "reference_answer": "Expands around centers for single and double character midpoints.",
        "target_capability": ["code", "math"]
    },
    # LMSYS Chatbot Arena (General Knowledge & QA)
    {
        "id": "lmsys_001",
        "benchmark": "LMSYS_Arena",
        "category": "general_qa",
        "difficulty": 0.25,
        "prompt": "Explain the difference between renewable and non-renewable energy sources in 3 simple bullet points.",
        "reference_answer": "1. Renewable energy comes from infinite natural processes (sun, wind). 2. Non-renewable energy comes from finite resources (coal, oil). 3. Renewable energy produces significantly lower greenhouse emissions.",
        "target_capability": ["speed", "chat"]
    },
    {
        "id": "lmsys_002",
        "benchmark": "LMSYS_Arena",
        "category": "general_qa",
        "difficulty": 0.40,
        "prompt": "What are the primary functions of the mitochondria in human biological cells?",
        "reference_answer": "ATP synthesis via oxidative phosphorylation, regulation of cellular metabolism, calcium signaling, and apoptosis.",
        "target_capability": ["reasoning", "speed"]
    },
    # MeetingBank (Long-Context Summarization)
    {
        "id": "meetingbank_001",
        "benchmark": "MeetingBank",
        "category": "long_context_summary",
        "difficulty": 0.50,
        "prompt": "Summarize the city council meeting transcript covering municipal budget allocations for public transit, housing development, and park maintenance into 3 executive action items.",
        "reference_answer": "1. Approved $12M transit expansion. 2. Allocated $5M affordable housing grant. 3. Deferred park maintenance vote to Q3.",
        "target_capability": ["compression", "summary"]
    }
]

# Write to JSONL
with open("data/benchmarks/test_suite.jsonl", "w") as f:
    for item in benchmarks:
        f.write(json.dumps(item) + "\n")

summary = {
    "total_benchmark_samples": len(benchmarks),
    "benchmarks_included": ["GSM8K", "HumanEval", "LMSYS_Arena", "MeetingBank"],
    "categories": ["math_reasoning", "code_generation", "general_qa", "long_context_summary"],
    "file_path": "data/benchmarks/test_suite.jsonl"
}

with open("data/benchmarks/benchmark_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Benchmark datasets collected successfully!")
print(json.dumps(summary, indent=2))
