"use client";

import React from "react";
import { Sliders, Zap, DollarSign, Clock, ShieldCheck, Database, Check } from "lucide-react";
import { SLAConfig, RoutingStrategy } from "@/types/gateway";

interface SLAControllerProps {
  slaConfig: SLAConfig;
  setSLAConfig: React.Dispatch<React.SetStateAction<SLAConfig>>;
}

export const SLAController: React.FC<SLAControllerProps> = ({ slaConfig, setSLAConfig }) => {
  const strategies: { id: RoutingStrategy; label: string; description: string; icon: string }[] = [
    {
      id: "BALANCED_OMNIROUTE",
      label: "Balanced OmniRoute (Default)",
      description: "Dual Lagrangian constrained optimization satisfying budget, latency, and quality simultaneously.",
      icon: "⚖️",
    },
    {
      id: "SPEED_FIRST",
      label: "Speed Priority (TTFT < 20ms)",
      description: "Prioritizes low latency vendor models and cache hits over cost savings.",
      icon: "⚡",
    },
    {
      id: "MAX_SAVINGS",
      label: "Maximum Budget Savings",
      description: "Aggressively routes to low-cost Flash models & token compression (-81%).",
      icon: "💰",
    },
    {
      id: "QUALITY_MAX",
      label: "Quality Target First",
      description: "Enforces strict reasoning capability thresholds for complex tasks.",
      icon: "🎯",
    },
  ];

  return (
    <div className="flex-1 p-6 space-y-6 overflow-y-auto bg-zinc-950/40">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-zinc-100 tracking-tight flex items-center gap-2">
          <Sliders className="h-5 w-5 text-indigo-400" />
          <span>Gateway SLA Governance & Routing Controller</span>
        </h1>
        <p className="text-xs text-zinc-500 font-mono mt-1">
          Dynamically configure latency caps, budget constraints, and caching thresholds on-the-fly.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Panel 1: SLA Target Sliders */}
        <div className="p-5 rounded-2xl glass-panel space-y-6">
          <h2 className="text-sm font-bold text-zinc-100 flex items-center gap-2 border-b border-zinc-800 pb-3">
            <Zap className="h-4 w-4 text-amber-400" />
            <span>SLA Constraints & Budget Targets</span>
          </h2>

          {/* Slider 1: Max Budget Target */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-zinc-300 font-medium flex items-center gap-1.5">
                <DollarSign className="h-3.5 w-3.5 text-emerald-400" />
                Max Budget Target ($ / 1k Tokens)
              </span>
              <span className="font-mono font-bold text-emerald-400 text-sm">
                ${slaConfig.maxBudgetTarget.toFixed(4)}
              </span>
            </div>
            <input
              type="range"
              min="0.0001"
              max="0.0050"
              step="0.0001"
              value={slaConfig.maxBudgetTarget}
              onChange={(e) =>
                setSLAConfig((prev) => ({ ...prev, maxBudgetTarget: parseFloat(e.target.value) }))
              }
              className="w-full accent-emerald-500 bg-zinc-800 rounded-lg cursor-pointer"
            />
            <p className="text-[11px] text-zinc-500 font-mono">
              Requests exceeding this target will trigger LLMLingua token compression or fallback to Flash models.
            </p>
          </div>

          {/* Slider 2: Latency Cap */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-zinc-300 font-medium flex items-center gap-1.5">
                <Clock className="h-3.5 w-3.5 text-indigo-400" />
                Latency SLA Cap (TTFT ms)
              </span>
              <span className="font-mono font-bold text-indigo-300 text-sm">
                {slaConfig.latencyCapMs} ms
              </span>
            </div>
            <input
              type="range"
              min="20"
              max="1000"
              step="10"
              value={slaConfig.latencyCapMs}
              onChange={(e) =>
                setSLAConfig((prev) => ({ ...prev, latencyCapMs: parseInt(e.target.value) }))
              }
              className="w-full accent-indigo-500 bg-zinc-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* Slider 3: Quality Threshold */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-zinc-300 font-medium flex items-center gap-1.5">
                <ShieldCheck className="h-3.5 w-3.5 text-purple-400" />
                Quality Score Target
              </span>
              <span className="font-mono font-bold text-purple-300 text-sm">
                {(slaConfig.qualityThreshold * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0.70"
              max="0.99"
              step="0.01"
              value={slaConfig.qualityThreshold}
              onChange={(e) =>
                setSLAConfig((prev) => ({ ...prev, qualityThreshold: parseFloat(e.target.value) }))
              }
              className="w-full accent-purple-500 bg-zinc-800 rounded-lg cursor-pointer"
            />
          </div>
        </div>

        {/* Panel 2: Strategy Selector & Cache Controls */}
        <div className="p-5 rounded-2xl glass-panel space-y-6">
          <h2 className="text-sm font-bold text-zinc-100 flex items-center gap-2 border-b border-zinc-800 pb-3">
            <Database className="h-4 w-4 text-cyan-400" />
            <span>Routing Strategy Presets</span>
          </h2>

          <div className="space-y-2.5">
            {strategies.map((strat) => {
              const isSelected = slaConfig.strategy === strat.id;
              return (
                <div
                  key={strat.id}
                  onClick={() => setSLAConfig((prev) => ({ ...prev, strategy: strat.id }))}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                    isSelected
                      ? "bg-indigo-600/15 border-indigo-500/40 text-zinc-100"
                      : "bg-zinc-900/40 border-zinc-800/80 hover:border-zinc-700 text-zinc-400"
                  }`}
                >
                  <div className="flex items-center justify-between font-semibold text-xs">
                    <span className="flex items-center gap-2 text-zinc-200">
                      <span>{strat.icon}</span>
                      <span>{strat.label}</span>
                    </span>
                    {isSelected && <Check className="h-4 w-4 text-indigo-400" />}
                  </div>
                  <p className="text-[11px] text-zinc-500 mt-1">{strat.description}</p>
                </div>
              );
            })}
          </div>

          {/* Cache Engine Toggles */}
          <div className="pt-2 border-t border-zinc-800 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-zinc-300 font-medium">Tier-0 Exact Redis Hash Cache</span>
              <input
                type="checkbox"
                checked={slaConfig.enableExactCache}
                onChange={(e) =>
                  setSLAConfig((prev) => ({ ...prev, enableExactCache: e.target.checked }))
                }
                className="h-4 w-4 accent-indigo-500 rounded cursor-pointer"
              />
            </div>

            <div className="flex items-center justify-between text-xs">
              <span className="text-zinc-300 font-medium">Tier-1 Semantic Vector Qdrant Cache</span>
              <input
                type="checkbox"
                checked={slaConfig.enableSemanticCache}
                onChange={(e) =>
                  setSLAConfig((prev) => ({ ...prev, enableSemanticCache: e.target.checked }))
                }
                className="h-4 w-4 accent-indigo-500 rounded cursor-pointer"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
