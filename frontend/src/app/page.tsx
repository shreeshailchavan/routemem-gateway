"use client";

import React, { useState, useEffect } from "react";
import { Navbar, ThemeMode, ColorPalette } from "@/components/layout/Navbar";
import { Sidebar, NavTab } from "@/components/layout/Sidebar";
import { ChatPlayground } from "@/components/playground/ChatPlayground";
import { PipelineInspector } from "@/components/playground/PipelineInspector";
import { TelemetryDashboard } from "@/components/telemetry/TelemetryDashboard";
import { SLAController } from "@/components/governance/SLAController";
import { AuditLogs } from "@/components/audit/AuditLogs";
import { ChatMessage, ExecutionTrace, SLAConfig } from "@/types/gateway";
import { initialChatMessages, initialSLAConfig, sampleExecutionTrace, sampleTelemetrySummary } from "@/lib/mockData";

export default function GatewayPage() {
  const [activeTab, setActiveTab] = useState<NavTab>("playground");
  const [slaConfig, setSLAConfig] = useState<SLAConfig>(initialSLAConfig);
  const [messages, setMessages] = useState<ChatMessage[]>(initialChatMessages);
  const [activeTrace, setActiveTrace] = useState<ExecutionTrace | undefined>(sampleExecutionTrace);

  // Theme & Palette State
  const [themeMode, setThemeMode] = useState<ThemeMode>("dark");
  const [colorPalette, setColorPalette] = useState<ColorPalette>("obsidian");

  useEffect(() => {
    const root = document.documentElement;
    // Apply light/dark mode class
    if (themeMode === "light") {
      root.classList.add("light");
      root.classList.remove("dark");
    } else {
      root.classList.add("dark");
      root.classList.remove("light");
    }

    // Apply color palette class
    root.classList.remove("palette-obsidian", "palette-emerald", "palette-gemini");
    root.classList.add(`palette-${colorPalette}`);
  }, [themeMode, colorPalette]);

  return (
    <div className={`flex flex-col min-h-screen ${themeMode === "light" ? "bg-slate-50 text-slate-900" : "bg-zinc-950 text-zinc-100"} font-sans transition-colors duration-200`}>
      {/* Top Navbar */}
      <Navbar
        slaConfig={slaConfig}
        themeMode={themeMode}
        setThemeMode={setThemeMode}
        colorPalette={colorPalette}
        setColorPalette={setColorPalette}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Navigation Sidebar */}
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

        {/* Dynamic Viewport */}
        {activeTab === "playground" && (
          <div className="flex-1 flex overflow-hidden">
            <ChatPlayground
              messages={messages}
              setMessages={setMessages}
              activeTrace={activeTrace}
              setActiveTrace={setActiveTrace}
              slaConfig={slaConfig}
            />
            <PipelineInspector trace={activeTrace} />
          </div>
        )}

        {activeTab === "telemetry" && (
          <TelemetryDashboard telemetry={sampleTelemetrySummary} />
        )}

        {activeTab === "governance" && (
          <SLAController slaConfig={slaConfig} setSLAConfig={setSLAConfig} />
        )}

        {activeTab === "audit" && (
          <AuditLogs setActiveTrace={setActiveTrace} setActiveTab={setActiveTab} />
        )}
      </div>
    </div>
  );
}
