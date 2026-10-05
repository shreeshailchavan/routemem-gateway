"use client";

import React from "react";
import { Cpu, Zap, ShieldCheck, Database, Sun, Moon, Palette } from "lucide-react";
import { SLAConfig } from "@/types/gateway";

export type ThemeMode = "dark" | "light";
export type ColorPalette = "obsidian" | "emerald" | "gemini";

interface NavbarProps {
  slaConfig: SLAConfig;
  themeMode: ThemeMode;
  setThemeMode: (mode: ThemeMode) => void;
  colorPalette: ColorPalette;
  setColorPalette: (palette: ColorPalette) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  slaConfig,
  themeMode,
  setThemeMode,
  colorPalette,
  setColorPalette,
}) => {
  return (
    <header className="h-16 border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur-xl sticky top-0 z-50 px-6 flex items-center justify-between transition-colors">
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

      {/* Live System Indicators & Theme Controls */}
      <div className="flex items-center space-x-3 text-xs font-mono">
        {/* Gateway Health Badge */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span className="font-medium">Gateway Active</span>
        </div>

        {/* Color Palette Selector Dropdown */}
        <div className="flex items-center space-x-1 px-2 py-1 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-300">
          <Palette className="h-3.5 w-3.5 text-indigo-400 mr-1" />
          <button
            onClick={() => setColorPalette("obsidian")}
            className={`px-2 py-0.5 rounded text-[11px] transition-all ${
              colorPalette === "obsidian"
                ? "bg-indigo-600 text-white font-bold"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            Obsidian
          </button>
          <button
            onClick={() => setColorPalette("emerald")}
            className={`px-2 py-0.5 rounded text-[11px] transition-all ${
              colorPalette === "emerald"
                ? "bg-emerald-600 text-white font-bold"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            Groq LPU
          </button>
          <button
            onClick={() => setColorPalette("gemini")}
            className={`px-2 py-0.5 rounded text-[11px] transition-all ${
              colorPalette === "gemini"
                ? "bg-cyan-600 text-white font-bold"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            Gemini
          </button>
        </div>

        {/* Light / Dark Mode Toggle */}
        <button
          onClick={() => setThemeMode(themeMode === "dark" ? "light" : "dark")}
          className="p-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 transition-all flex items-center gap-1.5"
          title="Toggle Light / Dark Mode"
        >
          {themeMode === "dark" ? (
            <>
              <Sun className="h-3.5 w-3.5 text-amber-400" />
              <span className="text-[11px]">Light</span>
            </>
          ) : (
            <>
              <Moon className="h-3.5 w-3.5 text-indigo-400" />
              <span className="text-[11px]">Dark</span>
            </>
          )}
        </button>

        {/* Security Shield */}
        <div className="hidden lg:flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-300">
          <ShieldCheck className="h-3.5 w-3.5 text-indigo-400" />
          <span className="font-semibold">GRPO Verified</span>
        </div>
      </div>
    </header>
  );
};
