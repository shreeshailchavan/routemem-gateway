export type CacheStatus = "EXACT_HIT" | "SEMANTIC_HIT" | "MISS";
export type RoutingStrategy = "BALANCED_OMNIROUTE" | "SPEED_FIRST" | "MAX_SAVINGS" | "QUALITY_MAX";
export type VendorProvider = "Google AI Studio" | "Groq LPU" | "vLLM Local" | "DeepSeek Cloud";

export interface PipelineStageTrace {
  stageId: number;
  name: string;
  status: "completed" | "skipped" | "active" | "error";
  latencyMs: number;
  details: string;
}

export interface ExecutionTrace {
  id: string;
  timestamp: string;
  query: string;
  cacheStatus: CacheStatus;
  exactHashHit: boolean;
  semanticCosineScore?: number;
  originalTokens: number;
  compressedTokens: number;
  compressionReductionPct: number;
  profilerDifficulty: number; // 0.0 - 1.0
  profilerDomain: string; // "Coding", "Math", "Reasoning", "General"
  selectedModel: string;
  selectedVendor: VendorProvider;
  ttftMs: number;
  totalLatencyMs: number;
  queryCost: number;
  baselineCost: number;
  savingsPct: number;
  pipelineStages: PipelineStageTrace[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  isStreaming?: boolean;
  trace?: ExecutionTrace;
}

export interface SLAConfig {
  maxBudgetTarget: number; // e.g. 0.001 USD / 1k tokens
  latencyCapMs: number; // e.g. 200 ms
  qualityThreshold: number; // 0.70 - 0.99
  strategy: RoutingStrategy;
  enableExactCache: boolean;
  enableSemanticCache: boolean;
  semanticSimilarityThreshold: number; // 0.80 - 0.98
}

export interface TelemetrySummary {
  totalQueries: number;
  exactHitsCount: number;
  semanticHitsCount: number;
  vendorDispatchesCount: number;
  totalSpendUSD: number;
  baselineSpendUSD: number;
  netSavingsUSD: number;
  netSavingsPct: number;
  avgTtftMs: number;
  avgCompressionRatioPct: number;
  slaCompliancePct: number;
}
