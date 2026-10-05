"use client";

import React, { useState } from "react";
import { FileText, Download, Search, CheckCircle2, Clock, DollarSign, ExternalLink } from "lucide-react";
import { ExecutionTrace } from "@/types/gateway";
import { sampleExecutionTrace, sampleExactHitTrace } from "@/lib/mockData";
import { formatCost, formatLatency } from "@/lib/utils";

interface AuditLogsProps {
  setActiveTrace: (trace: ExecutionTrace) => void;
  setActiveTab: (tab: any) => void;
}

export const AuditLogs: React.FC<AuditLogsProps> = ({ setActiveTrace, setActiveTab }) => {
  const [searchTerm, setSearchTerm] = useState("");

  const mockTraces: ExecutionTrace[] = [
    sampleExecutionTrace,
    sampleExactHitTrace,
    {
      ...sampleExecutionTrace,
      id: "trc_b881c20a",
      timestamp: "19:42:10",
      query: "Compare Qdrant vs Milvus vector database latency benchmarks",
      cacheStatus: "SEMANTIC_HIT",
      semanticCosineScore: 0.94,
      originalTokens: 820,
      compressedTokens: 210,
      selectedModel: "gpt-oss-120b",
      selectedVendor: "Groq LPU",
      ttftMs: 14.20,
      totalLatencyMs: 142.10,
      queryCost: 0.00015,
      savingsPct: 99.4,
    },
    {
      ...sampleExecutionTrace,
      id: "trc_c993d31b",
      timestamp: "19:38:05",
      query: "Explain GRPO alignment for reasoning models",
      cacheStatus: "MISS",
      originalTokens: 1100,
      compressedTokens: 240,
      selectedModel: "qwen-27b",
      selectedVendor: "Groq LPU",
      ttftMs: 22.40,
      totalLatencyMs: 210.00,
      queryCost: 0.00018,
      savingsPct: 99.3,
    },
  ];

  const filteredTraces = mockTraces.filter((t) =>
    t.query.toLowerCase().includes(searchTerm.toLowerCase()) ||
    t.selectedModel.toLowerCase().includes(searchTerm.toLowerCase()) ||
    t.cacheStatus.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const exportLogsAsJSON = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(mockTraces, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "routemem_audit_traces.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="flex-1 p-6 space-y-6 overflow-y-auto bg-zinc-950/40">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between md:items-center gap-4">
        <div>
          <h1 className="text-xl font-bold text-zinc-100 tracking-tight flex items-center gap-2">
            <FileText className="h-5 w-5 text-indigo-400" />
            <span>Audit & Execution Trace Logs</span>
          </h1>
          <p className="text-xs text-zinc-500 font-mono mt-1">
            Filterable request traces and empirical benchmark exporter for IEEE / Scopus replication.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={exportLogsAsJSON}
            className="px-4 py-2 rounded-xl bg-zinc-900 hover:bg-zinc-800 text-zinc-200 border border-zinc-800 font-mono text-xs flex items-center space-x-2 transition-all shadow-sm"
          >
            <Download className="h-4 w-4 text-indigo-400" />
            <span>Export Traces (JSON)</span>
          </button>
        </div>
      </div>

      {/* Search Input */}
      <div className="relative">
        <Search className="absolute left-3.5 top-3 h-4 w-4 text-zinc-500" />
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Filter audit logs by query content, model, or cache status..."
          className="w-full bg-zinc-900/80 border border-zinc-800 rounded-xl pl-10 pr-4 py-2.5 text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-indigo-500/60 font-mono"
        />
      </div>

      {/* Audit Log Table */}
      <div className="rounded-2xl glass-panel overflow-hidden border border-zinc-800/80">
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead className="bg-zinc-900/80 text-zinc-400 uppercase text-[10px] tracking-wider border-b border-zinc-800">
              <tr>
                <th className="p-3.5">Trace ID & Time</th>
                <th className="p-3.5">Query Summary</th>
                <th className="p-3.5">Cache Status</th>
                <th className="p-3.5">Selected Model</th>
                <th className="p-3.5">TTFT Latency</th>
                <th className="p-3.5">Cost ($)</th>
                <th className="p-3.5 text-right">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/60 text-zinc-300">
              {filteredTraces.map((trace) => {
                const isExact = trace.cacheStatus === "EXACT_HIT";
                const isSemantic = trace.cacheStatus === "SEMANTIC_HIT";

                return (
                  <tr key={trace.id} className="hover:bg-zinc-900/60 transition-colors">
                    <td className="p-3.5 text-zinc-400">
                      <div className="font-semibold text-zinc-200">{trace.id}</div>
                      <div className="text-[10px] text-zinc-500">{trace.timestamp}</div>
                    </td>

                    <td className="p-3.5 max-w-xs truncate text-zinc-200 font-sans">
                      {trace.query}
                    </td>

                    <td className="p-3.5">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                          isExact
                            ? "glow-exact-cache"
                            : isSemantic
                            ? "glow-semantic-cache"
                            : "bg-indigo-500/10 text-indigo-400 border-indigo-500/20"
                        }`}
                      >
                        {trace.cacheStatus}
                      </span>
                    </td>

                    <td className="p-3.5">
                      <span className="font-semibold text-indigo-300">{trace.selectedModel}</span>
                      <div className="text-[10px] text-zinc-500">{trace.selectedVendor}</div>
                    </td>

                    <td className="p-3.5 font-bold text-zinc-200">
                      {formatLatency(trace.ttftMs)}
                    </td>

                    <td className="p-3.5 font-bold text-emerald-400">
                      {formatCost(trace.queryCost)}
                    </td>

                    <td className="p-3.5 text-right">
                      <button
                        onClick={() => {
                          setActiveTrace(trace);
                          setActiveTab("playground");
                        }}
                        className="px-2.5 py-1 rounded bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 text-[11px] flex items-center space-x-1 ml-auto transition-all"
                      >
                        <span>Inspect</span>
                        <ExternalLink className="h-3 w-3" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
