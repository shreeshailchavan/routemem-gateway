"use client";

import React from "react";
import { CheckCircle2, Clock, Zap, Cpu, Database, Shrink, Brain, DollarSign, Activity } from "lucide-react";
import { ExecutionTrace } from "@/types/gateway";
import { formatCost, formatLatency } from "@/lib/utils";

interface PipelineInspectorProps {
  trace?: ExecutionTrace;
}

export const PipelineInspector: React.FC<PipelineInspectorProps> = ({ trace }) => {
  if (!trace) {
    return (
      <aside className="w-80 border-l border-zinc-800/80 bg-zinc-950/60 backdrop-blur-md p-5 flex flex-col justify-center items-center text-center">
        <Activity className="h-10 w-10 text-zinc-700 animate-pulse mb-3" />
        <h3 className="text-sm font-medium text-zinc-400">No Active Trace</h3>
        <p className="text-xs text-zinc-600 mt-1 max-w-[200px]">
          Submit a prompt in the Chat Studio to observe the live 8-stage execution pipeline.
        </p>
      </aside>
    );
  }

  const isExactHit = trace.cacheStatus === "EXACT_HIT";
  const isSemanticHit = trace.cacheStatus === "SEMANTIC_HIT";

  return (
    <aside className="w-96 border-l border-zinc-800/80 bg-zinc-950/80 backdrop-blur-xl p-5 overflow-y-auto space-y-6 shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-zinc-800">
        <div>
          <h2 className="text-sm font-bold text-zinc-100 flex items-center gap-2">
            <Cpu className="h-4 w-4 text-indigo-400" />
            <span>8-Stage Pipeline Trace</span>
          </h2>
          <p className="text-[11px] font-mono text-zinc-500 mt-0.5">Trace ID: {trace.id}</p>
        </div>
        <span
          className={`px-2.5 py-1 rounded-full text-xs font-mono font-bold border ${
            isExactHit
              ? "glow-exact-cache"
              : isSemanticHit
              ? "glow-semantic-cache"
              : "bg-indigo-500/10 text-indigo-400 border-indigo-500/30"
          }`}
        >
          {trace.cacheStatus}
        </span>
      </div>

      {/* KPI Cards Summary */}
      <div className="grid grid-cols-2 gap-2.5 font-mono">
        <div className="p-3 rounded-xl glass-panel space-y-1">
          <div className="text-[10px] text-zinc-500 flex items-center gap-1">
            <Clock className="h-3 w-3 text-indigo-400" />
            <span>TTFT Latency</span>
          </div>
          <p className="text-base font-bold text-zinc-100">{formatLatency(trace.ttftMs)}</p>
        </div>

        <div className="p-3 rounded-xl glass-panel space-y-1">
          <div className="text-[10px] text-zinc-500 flex items-center gap-1">
            <DollarSign className="h-3 w-3 text-emerald-400" />
            <span>Request Cost</span>
          </div>
          <p className="text-base font-bold text-emerald-400">{formatCost(trace.queryCost)}</p>
        </div>

        <div className="p-3 rounded-xl glass-panel space-y-1">
          <div className="text-[10px] text-zinc-500 flex items-center gap-1">
            <Shrink className="h-3 w-3 text-purple-400" />
            <span>Compression</span>
          </div>
          <p className="text-base font-bold text-purple-300">-{trace.compressionReductionPct.toFixed(1)}%</p>
        </div>

        <div className="p-3 rounded-xl glass-panel space-y-1">
          <div className="text-[10px] text-zinc-500 flex items-center gap-1">
            <Brain className="h-3 w-3 text-cyan-400" />
            <span>Difficulty Score</span>
          </div>
          <p className="text-base font-bold text-cyan-300">{(trace.profilerDifficulty * 100).toFixed(0)} / 100</p>
        </div>
      </div>

      {/* Selected Model Dispatch Badge */}
      <div className="p-3.5 rounded-xl bg-gradient-to-r from-indigo-950/40 via-zinc-900 to-zinc-900 border border-indigo-500/30 space-y-2">
        <div className="flex justify-between items-center text-xs">
          <span className="text-zinc-400">Selected Vendor / Model</span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            {trace.savingsPct.toFixed(1)}% Saved
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="font-bold text-sm text-indigo-200">{trace.selectedModel}</span>
          <span className="text-xs font-mono text-zinc-400">{trace.selectedVendor}</span>
        </div>
      </div>

      {/* Vertical Stage Timeline */}
      <div className="space-y-4 pt-2">
        <h3 className="text-xs font-mono font-semibold text-zinc-400 uppercase tracking-wider">
          Stage Execution Flow
        </h3>

        <div className="relative pl-6 space-y-5 before:absolute before:left-2.5 before:top-2 before:bottom-2 before:w-[2px] before:bg-zinc-800">
          {trace.pipelineStages.map((stage) => {
            const isCompleted = stage.status === "completed";
            const isSkipped = stage.status === "skipped";

            return (
              <div key={stage.stageId} className="relative group">
                {/* Timeline Dot */}
                <span
                  className={`absolute -left-[19px] top-0.5 h-3.5 w-3.5 rounded-full border-2 flex items-center justify-center transition-all ${
                    isCompleted
                      ? "bg-emerald-500 border-emerald-400 text-zinc-950 shadow-sm shadow-emerald-500/50"
                      : isSkipped
                      ? "bg-zinc-800 border-zinc-700 text-zinc-500"
                      : "bg-indigo-500 border-indigo-400"
                  }`}
                >
                  {isCompleted && <CheckCircle2 className="h-2.5 w-2.5 text-zinc-950" />}
                </span>

                {/* Stage Detail Card */}
                <div
                  className={`p-3 rounded-xl border text-xs transition-all ${
                    isCompleted
                      ? "bg-zinc-900/60 border-zinc-800 hover:border-zinc-700 text-zinc-200"
                      : "bg-zinc-950/40 border-zinc-900 text-zinc-600"
                  }`}
                >
                  <div className="flex items-center justify-between font-medium">
                    <span>{stage.name}</span>
                    <span className="font-mono text-[10px] text-zinc-400">
                      {stage.latencyMs > 0 ? formatLatency(stage.latencyMs) : "0 ms"}
                    </span>
                  </div>
                  <p className="text-[11px] text-zinc-400 mt-1 font-mono leading-relaxed">
                    {stage.details}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </aside>
  );
};
