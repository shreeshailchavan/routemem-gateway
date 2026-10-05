"use client";

import React, { useState } from "react";
import { Navbar } from "@/components/layout/Navbar";
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

  return (
    <div className="flex flex-col min-h-screen bg-zinc-950 text-zinc-100 font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      {/* Top Navbar */}
      <Navbar slaConfig={slaConfig} />

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
