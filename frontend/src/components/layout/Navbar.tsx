"use client";

import React from "react";
import { Cpu, Zap, Activity, ShieldCheck, Database } from "lucide-react";
import { SLAConfig } from "@/types/gateway";

interface NavbarProps {
  slaConfig: SLAConfig;
}

export const Navbar: React.FC<NavbarProps> = ({ slaConfig }) => {
  return (
    <header className="h-16 border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur-xl sticky top-0 z-50 px-6 flex items-center justify-between">
      {/* Brand Title */}
      <div className="flex items-center space-x-3">
        <div className="h-9 w-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-emerald-500 p-[1px] shadow-lg shadow-indigo-500/20">
          <div className="h-full w-full bg-zinc-950 rounded-[11px] flex items-center justify-center">
            <Cpu className="h-5 w-5 text-indigo-400" />
          </div>
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-zinc-100 via-zinc-200 to-zinc-400 bg-clip-text text-transparent">
              RouteMem
            </span>
            <span className="px-2 py-0.5 text-[10px] font-mono font-semibold rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              v1.2 AI Gateway
            </span>
          </div>
          <p className="text-[11px] text-zinc-500 font-mono">
            8-Stage Budget-Constrained Multi-LLM Router
          </p>
        </div>
      </div>

      {/* Live System Indicators */}
      <div className="flex items-center space-x-4 text-xs font-mono">
        {/* Gateway Health Badge */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span className="font-medium">Gateway Active</span>
        </div>

        {/* Cache Engine Badge */}
        <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-300">
          <Database className="h-3.5 w-3.5 text-cyan-400" />
          <span>Tier-0 Redis & Tier-1 Qdrant</span>
        </div>

        {/* SLA Mode Badge */}
        <div className="hidden lg:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-300">
          <Zap className="h-3.5 w-3.5 text-amber-400" />
          <span>Max SLA: {slaConfig.latencyCapMs}ms | ${slaConfig.maxBudgetTarget}/1k</span>
        </div>

        {/* Security Shield */}
        <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-300">
          <ShieldCheck className="h-3.5 w-3.5 text-indigo-400" />
          <span className="font-semibold">GRPO Verified</span>
        </div>
      </div>
    </header>
  );
};
