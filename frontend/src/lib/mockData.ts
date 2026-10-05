import { ChatMessage, ExecutionTrace, SLAConfig, TelemetrySummary } from "@/types/gateway";

export const initialSLAConfig: SLAConfig = {
  maxBudgetTarget: 0.0005,
  latencyCapMs: 200,
  qualityThreshold: 0.88,
  strategy: "BALANCED_OMNIROUTE",
  enableExactCache: true,
  enableSemanticCache: true,
  semanticSimilarityThreshold: 0.92,
};

export const sampleExecutionTrace: ExecutionTrace = {
  id: "trc_9942a1f0",
  timestamp: new Date().toLocaleTimeString(),
  query: "Implement dynamic programming solution for 0/1 Knapsack in Rust with unit tests",
  cacheStatus: "MISS",
  exactHashHit: false,
  semanticCosineScore: 0.74,
  originalTokens: 1450,
  compressedTokens: 272,
  compressionReductionPct: 81.24,
  profilerDifficulty: 0.78,
  profilerDomain: "Coding / Algorithms",
  selectedModel: "gemini-3.8-flash",
  selectedVendor: "Google AI Studio",
  ttftMs: 14.20,
  totalLatencyMs: 182.40,
  queryCost: 0.00021,
  baselineCost: 0.04350,
  savingsPct: 99.52,
  pipelineStages: [
    { stageId: 0, name: "Exact SHA-256 Cache", status: "completed", latencyMs: 0.66, details: "Tier-0 Redis SHA-256 lookup miss" },
    { stageId: 1, name: "Semantic Vector Cache", status: "completed", latencyMs: 14.20, details: "Qdrant Cosine Similarity score 0.74 < threshold 0.92" },
    { stageId: 4, name: "LLMLingua-2 Compressor", status: "completed", latencyMs: 1.85, details: "Compressed 1,450 tokens -> 272 tokens (-81.2%)" },
    { stageId: 5, name: "INT8 DeBERTa Profiler", status: "completed", latencyMs: 0.24, details: "Intent: Coding/Algorithms | Difficulty: 0.78 (High)" },
    { stageId: 6, name: "UniRoute 4D Solver", status: "completed", latencyMs: 0.12, details: "OmniRouter dual Lagrangian selected gemini-3.8-flash (Budget < $0.0005)" },
    { stageId: 7, name: "Google AI Studio Dispatch", status: "completed", latencyMs: 165.33, details: "Streaming 420 response tokens @ 110 tok/sec" },
  ],
};

export const sampleExactHitTrace: ExecutionTrace = {
  id: "trc_exact_7f1",
  timestamp: new Date().toLocaleTimeString(),
  query: "What is the formula for Euler's Identity?",
  cacheStatus: "EXACT_HIT",
  exactHashHit: true,
  originalTokens: 14,
  compressedTokens: 14,
  compressionReductionPct: 0.0,
  profilerDifficulty: 0.05,
  profilerDomain: "General / Math",
  selectedModel: "Redis Hash Cache",
  selectedVendor: "Google AI Studio",
  ttftMs: 0.66,
  totalLatencyMs: 0.66,
  queryCost: 0.0000,
  baselineCost: 0.00042,
  savingsPct: 100.0,
  pipelineStages: [
    { stageId: 0, name: "Exact SHA-256 Cache", status: "completed", latencyMs: 0.66, details: "Tier-0 Redis SHA-256 Hash HIT. Returned cached response instantly." },
    { stageId: 1, name: "Semantic Vector Cache", status: "skipped", latencyMs: 0.0, details: "Bypassed due to Tier-0 exact match" },
    { stageId: 4, name: "LLMLingua-2 Compressor", status: "skipped", latencyMs: 0.0, details: "Bypassed" },
    { stageId: 5, name: "INT8 DeBERTa Profiler", status: "skipped", latencyMs: 0.0, details: "Bypassed" },
    { stageId: 6, name: "UniRoute 4D Solver", status: "skipped", latencyMs: 0.0, details: "Bypassed" },
    { stageId: 7, name: "Vendor Dispatch", status: "skipped", latencyMs: 0.0, details: "Bypassed (Served from Redis)" },
  ],
};

export const initialChatMessages: ChatMessage[] = [
  {
    id: "msg_welcome",
    role: "assistant",
    content: "Welcome to **RouteMem AI Gateway**! I am connected to our 8-stage dynamic routing engine. Send a prompt to observe real-time model selection, cache hits, token compression, and cost optimization.",
    timestamp: new Date().toLocaleTimeString(),
  },
  {
    id: "msg_user_1",
    role: "user",
    content: "Implement dynamic programming solution for 0/1 Knapsack in Rust with unit tests",
    timestamp: new Date().toLocaleTimeString(),
  },
  {
    id: "msg_assistant_1",
    role: "assistant",
    content: "Here is an optimized 0/1 Knapsack dynamic programming implementation in Rust:\n\n```rust\npub fn knapsack(weights: &[usize], values: &[usize], capacity: usize) -> usize {\n    let n = weights.len();\n    let mut dp = vec![0; capacity + 1];\n\n    for i in 0..n {\n        for w in (weights[i]..=capacity).rev() {\n            dp[w] = dp[w].max(dp[w - weights[i]] + values[i]);\n        }\n    }\n    dp[capacity]\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n    fn test_knapsack_basic() {\n        let weights = vec![2, 3, 4, 5];\n        let values = vec![3, 4, 5, 6];\n        assert_eq!(knapsack(&weights, &values, 5), 7);\n    }\n}\n```",
    timestamp: new Date().toLocaleTimeString(),
    trace: sampleExecutionTrace,
  },
];

export const sampleTelemetrySummary: TelemetrySummary = {
  totalQueries: 52410,
  exactHitsCount: 18420,
  semanticHitsCount: 14210,
  vendorDispatchesCount: 19780,
  totalSpendUSD: 4.12,
  baselineSpendUSD: 544.60,
  netSavingsUSD: 540.48,
  netSavingsPct: 99.24,
  avgTtftMs: 14.20,
  avgCompressionRatioPct: 81.20,
  slaCompliancePct: 99.85,
};
