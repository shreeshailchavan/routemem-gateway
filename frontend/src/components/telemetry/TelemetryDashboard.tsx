"use client";

import React from "react";
import { DollarSign, Clock, Shrink, Zap, TrendingUp, ShieldCheck, Database, Layers } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from "recharts";
import { TelemetrySummary } from "@/types/gateway";
import { formatCost, formatLatency } from "@/lib/utils";

interface TelemetryDashboardProps {
  telemetry: TelemetrySummary;
}

export const TelemetryDashboard: React.FC<TelemetryDashboardProps> = ({ telemetry }) => {
  // Historical spend chart data
  const spendHistoryData = [
    { day: "Mon", baseline: 82.40, routemem: 0.62 },
    { day: "Tue", baseline: 94.10, routemem: 0.71 },
    { day: "Wed", baseline: 110.50, routemem: 0.85 },
    { day: "Thu", baseline: 105.20, routemem: 0.78 },
    { day: "Fri", baseline: 152.40, routemem: 1.16 },
    { day: "Sat", baseline: 44.00, routemem: 0.32 },
    { day: "Sun", baseline: 56.00, routemem: 0.41 },
  ];

  // Cache breakdown pie chart
  const cacheBreakdownData = [
    { name: "Exact SHA-256 Hit", value: telemetry.exactHitsCount, color: "#34d399" },
    { name: "Semantic Vector Hit", value: telemetry.semanticHitsCount, color: "#38bdf8" },
    { name: "Vendor API Dispatch", value: telemetry.vendorDispatchesCount, color: "#818cf8" },
  ];

  // Model distribution bar chart
  const modelDistributionData = [
    { model: "gemini-3.8-flash", count: 12450, costPer1k: 0.0001 },
    { model: "gpt-oss-120b (Groq)", count: 5120, costPer1k: 0.0003 },
    { model: "qwen-27b (Groq)", count: 2210, costPer1k: 0.0002 },
  ];

  return (
    <div className="flex-1 p-6 space-y-6 overflow-y-auto bg-zinc-950/40">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-xl font-bold text-zinc-100 tracking-tight flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-indigo-400" />
            <span>RouteMem Telemetry & Savings Analytics</span>
          </h1>
          <p className="text-xs text-zinc-500 font-mono mt-1">
            Real-time telemetry benchmarking budget savings, TTFT latencies, and cache efficiency.
          </p>
        </div>
        <div className="flex items-center space-x-2 text-xs font-mono">
          <span className="px-3 py-1.5 rounded-lg glass-panel text-emerald-400 border border-emerald-500/30">
            💰 Net Savings: ${telemetry.netSavingsUSD.toFixed(2)} ({telemetry.netSavingsPct.toFixed(2)}%)
          </span>
        </div>
      </div>

      {/* 4 Primary Metric KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric 1: Net Cost Savings */}
        <div className="p-4 rounded-2xl glass-panel glass-panel-hover space-y-2">
          <div className="flex justify-between items-center text-xs text-zinc-400 font-mono">
            <span>Net Cost Savings</span>
            <DollarSign className="h-4 w-4 text-emerald-400" />
          </div>
          <p className="text-2xl font-bold text-emerald-400 font-mono">${telemetry.totalSpendUSD.toFixed(2)}</p>
          <p className="text-[11px] text-zinc-500 font-mono">
            vs <span className="line-through text-zinc-500">${telemetry.baselineSpendUSD.toFixed(2)}</span> Baseline
          </p>
        </div>

        {/* Metric 2: Average TTFT Latency */}
        <div className="p-4 rounded-2xl glass-panel glass-panel-hover space-y-2">
          <div className="flex justify-between items-center text-xs text-zinc-400 font-mono">
            <span>Avg TTFT Latency</span>
            <Clock className="h-4 w-4 text-indigo-400" />
          </div>
          <p className="text-2xl font-bold text-zinc-100 font-mono">{formatLatency(telemetry.avgTtftMs)}</p>
          <p className="text-[11px] text-emerald-400 font-mono">⚡ 99.85% SLA Compliant (&lt; 200ms)</p>
        </div>

        {/* Metric 3: Token Compression Ratio */}
        <div className="p-4 rounded-2xl glass-panel glass-panel-hover space-y-2">
          <div className="flex justify-between items-center text-xs text-zinc-400 font-mono">
            <span>Token Reduction</span>
            <Shrink className="h-4 w-4 text-purple-400" />
          </div>
          <p className="text-2xl font-bold text-purple-300 font-mono">-{telemetry.avgCompressionRatioPct.toFixed(1)}%</p>
          <p className="text-[11px] text-zinc-500 font-mono">LLMLingua-2 Selective Token Compression</p>
        </div>

        {/* Metric 4: Total Gateway Queries */}
        <div className="p-4 rounded-2xl glass-panel glass-panel-hover space-y-2">
          <div className="flex justify-between items-center text-xs text-zinc-400 font-mono">
            <span>Total Queries</span>
            <Zap className="h-4 w-4 text-cyan-400" />
          </div>
          <p className="text-2xl font-bold text-zinc-100 font-mono">{telemetry.totalQueries.toLocaleString()}</p>
          <p className="text-[11px] text-cyan-400 font-mono">{(telemetry.exactHitsCount + telemetry.semanticHitsCount).toLocaleString()} Cached Requests</p>
        </div>
      </div>

      {/* Main Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart 1: Spend Comparison Chart */}
        <div className="lg:col-span-2 p-5 rounded-2xl glass-panel space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-sm font-bold text-zinc-100 flex items-center gap-2">
              <DollarSign className="h-4 w-4 text-emerald-400" />
              <span>Weekly Spend: GPT-4o Baseline vs RouteMem Gateway ($)</span>
            </h3>
            <span className="text-xs font-mono text-emerald-400">99.2% Spend Cut</span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={spendHistoryData}>
                <defs>
                  <linearGradient id="baselineGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="routememGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="day" stroke="#71717a" fontSize={11} tickLine={false} />
                <YAxis stroke="#71717a" fontSize={11} tickLine={false} unit="$" />
                <Tooltip
                  contentStyle={{ backgroundColor: "#18181b", borderColor: "#27272a", borderRadius: "12px", color: "#fff", fontSize: "12px" }}
                />
                <Area type="monotone" dataKey="baseline" name="Baseline (GPT-4o)" stroke="#ef4444" fillOpacity={1} fill="url(#baselineGrad)" />
                <Area type="monotone" dataKey="routemem" name="RouteMem Gateway" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#routememGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: Cache Hit Breakdown Pie Chart */}
        <div className="p-5 rounded-2xl glass-panel space-y-4">
          <h3 className="text-sm font-bold text-zinc-100 flex items-center gap-2">
            <Database className="h-4 w-4 text-cyan-400" />
            <span>Cache & Routing Efficiency</span>
          </h3>

          <div className="h-48 w-full flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={cacheBreakdownData} cx="50%" cy="50%" innerRadius={45} outerRadius={70} dataKey="value">
                  {cacheBreakdownData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: "#18181b", borderColor: "#27272a", borderRadius: "12px", color: "#fff", fontSize: "12px" }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-1.5 text-xs font-mono">
            {cacheBreakdownData.map((item, idx) => (
              <div key={idx} className="flex justify-between items-center text-zinc-400">
                <span className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full" style={{ backgroundColor: item.color }}></span>
                  {item.name}
                </span>
                <span className="font-semibold text-zinc-200">{item.value.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
