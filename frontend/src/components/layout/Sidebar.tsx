"use client";

import React from "react";
import { MessageSquare, BarChart3, Sliders, FileText, Layers, Sparkles } from "lucide-react";

export type NavTab = "playground" | "telemetry" | "governance" | "audit";

interface SidebarProps {
  activeTab: NavTab;
  setActiveTab: (tab: NavTab) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab }) => {
  const navItems = [
    { id: "playground", label: "Chat Studio", icon: MessageSquare, badge: "Live Trace" },
    { id: "telemetry", label: "Analytics & Telemetry", icon: BarChart3, badge: "99.9% Save" },
    { id: "governance", label: "SLA Governance", icon: Sliders, badge: "Dynamic" },
    { id: "audit", label: "Audit & Trace Logs", icon: FileText, badge: "CSV/JSON" },
  ];

  return (
    <aside className="w-64 border-r border-zinc-800/80 bg-zinc-950/60 backdrop-blur-md p-4 flex flex-col justify-between shrink-0">
      <div className="space-y-6">
        <div>
          <p className="px-3 text-[11px] font-mono font-semibold text-zinc-500 uppercase tracking-wider mb-3">
            Gateway Navigation
          </p>
          <nav className="space-y-1.5">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id as NavTab)}
                  className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? "bg-gradient-to-r from-indigo-500/15 via-indigo-500/10 to-transparent border border-indigo-500/30 text-indigo-200 shadow-sm shadow-indigo-500/10"
                      : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/60 hover:border-zinc-800 border border-transparent"
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <Icon className={`h-4 w-4 ${isActive ? "text-indigo-400" : "text-zinc-500"}`} />
                    <span>{item.label}</span>
                  </div>
                  {item.badge && (
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded-md ${
                        isActive
                          ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
                          : "bg-zinc-900 text-zinc-500 border border-zinc-800"
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Vendor Availability Catalog Card */}
        <div className="p-3.5 rounded-xl glass-panel space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold text-zinc-300">
            <span className="flex items-center space-x-2">
              <Layers className="h-4 w-4 text-purple-400" />
              <span>Multi-Vendor Pool</span>
            </span>
            <span className="text-[10px] font-mono text-zinc-500">18 Models</span>
          </div>
          <div className="space-y-1.5 text-[11px] font-mono">
            <div className="flex justify-between items-center text-zinc-400">
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400"></span>
                Google Gemini 2.5/Pro
              </span>
              <span className="text-emerald-400 font-semibold">Free/Tier</span>
            </div>
            <div className="flex justify-between items-center text-zinc-400">
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400"></span>
                Groq LPU (Llama/Qwen)
              </span>
              <span className="text-emerald-400 font-semibold">Free LPU</span>
            </div>
            <div className="flex justify-between items-center text-zinc-400">
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-indigo-400"></span>
                OpenAI (o1/o3/gpt-4o)
              </span>
              <span className="text-indigo-400 font-semibold">Premium</span>
            </div>
            <div className="flex justify-between items-center text-zinc-400">
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-amber-400"></span>
                Claude 3.5/3.7/Opus
              </span>
              <span className="text-amber-400 font-semibold">Premium</span>
            </div>
            <div className="flex justify-between items-center text-zinc-400">
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-cyan-400"></span>
                Mistral Large/Coder
              </span>
              <span className="text-cyan-400 font-semibold">Premium</span>
            </div>
          </div>
        </div>
      </div>

      {/* Powered by UniRoute footer */}
      <div className="p-3 rounded-xl bg-zinc-900/40 border border-zinc-800/80 text-[11px] text-zinc-400 space-y-1">
        <div className="flex items-center space-x-1.5 text-indigo-400 font-medium">
          <Sparkles className="h-3.5 w-3.5" />
          <span>UniRoute 4D Optim</span>
        </div>
        <p className="text-zinc-500 leading-tight">
          Dual Lagrangian solver optimizing quality, cost, and latency across 18 models.
        </p>
      </div>
    </aside>
  );
};
