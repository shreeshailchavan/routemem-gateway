import time
import numpy as np
from typing import Dict, List, Tuple
from app.cache.dense_embedder import DenseEmbedder
from app.router.profiler import QueryProfiler

# 1. Define Representative Anchor Prompts for Intent Centroids
ANCHORS = {
    "complex_reasoning": [
        "A farmer needs to cross a river with a wolf, a goat, and a cabbage. The boat can only carry the farmer and one item.",
        "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have in total?",
        "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Prove formal deductive validity.",
        "Solve this logic puzzle: Three people check into a hotel room that costs thirty dollars. They each contribute ten dollars.",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "Explain the counter-intuitive Monty Hall problem and calculate conditional probability using Bayes theorem."
    ],
    "domain_expert": [
        "Analyze the antitrust implications of bundled SaaS pricing under Section 2 of the Sherman Act and Clayton Act.",
        "What are the differential diagnoses for acute intermittent porphyria presenting with severe abdominal pain and neuropathy?",
        "Explain the macroeconomic transmission mechanism of quantitative tightening on sovereign yield curve inversion.",
        "Derive the master equation for decoherence in an open quantum system interacting with a thermal reservoir.",
        "Examine the legal doctrine of forum non conveniens in cross-border intellectual property infringement litigation."
    ],
    "code_generation": [
        "Implement an asynchronous lock-free concurrent hash map in C++ using atomic compare-and-swap and hazard pointers.",
        "Write a distributed consensus algorithm adhering to Raft protocol with leader election and log replication in Go.",
        "Write a Python script to optimize an AVL tree with O(log n) self-balancing rotations and deletion operations.",
        "Design a high-throughput SQL database schema with composite indexing, foreign key constraints, and partition pruning.",
        "Write a GPU CUDA kernel to compute matrix multiplication using shared memory tiling and vectorized memory loads."
    ],
    "simple_qa": [
        "Hello, how are you today?",
        "What is the capital city of France?",
        "What is a def in python and how do I write a class? Write a beginner hello world.",
        "How do I import math in python?",
        "Who was the first president of the United States?",
        "What is the freezing point of water in Celsius?"
    ]
}

def generate_centroids(embedder: DenseEmbedder) -> Dict[str, np.ndarray]:
    """Generates unit-normalized 384-dim centroids for each intent category."""
    centroids = {}
    for intent, prompts in ANCHORS.items():
        vecs = [np.array(embedder.embed(p), dtype=np.float32) for p in prompts]
        mean_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        centroids[intent] = mean_vec / norm if norm > 0 else mean_vec
    return centroids

if __name__ == "__main__":
    embedder = DenseEmbedder()
    print("Generating calibrated prototype centroids...")
    centroids = generate_centroids(embedder)
    for intent, vec in centroids.items():
        print(f"Centroid '{intent}': shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")
