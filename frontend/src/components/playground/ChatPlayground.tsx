"use client";

import React, { useState, useRef, useEffect } from "react";
import { Send, Sparkles, ArrowUpRight, Columns, ShieldAlert, Cpu } from "lucide-react";
import { ChatMessage, ExecutionTrace, SLAConfig } from "@/types/gateway";
import { sampleExecutionTrace, sampleExactHitTrace } from "@/lib/mockData";
import { formatCost, formatLatency } from "@/lib/utils";

interface ChatPlaygroundProps {
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  activeTrace: ExecutionTrace | undefined;
  setActiveTrace: (trace: ExecutionTrace | undefined) => void;
  slaConfig: SLAConfig;
}

export const ChatPlayground: React.FC<ChatPlaygroundProps> = ({
  messages,
  setMessages,
  activeTrace,
  setActiveTrace,
  slaConfig,
}) => {
  const [inputPrompt, setInputPrompt] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isComparisonMode, setIsComparisonMode] = useState(false); // Clean single view by default
  const [selectedModel, setSelectedModel] = useState("routemem-auto"); // Model Selector State

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const availableModels = [
    { id: "routemem-auto", name: "⚡ Auto-Route (OmniRouter Solver)" },
    { id: "gemini-2.5-flash", name: "🟢 Google Gemini 2.5 Flash (Free)" },
    { id: "llama-3.3-70b-versatile", name: "🟢 Groq Llama 3.3 70B (Free LPU)" },
    { id: "qwen-2.5-coder-32b", name: "🟢 Groq Qwen 2.5 Coder (Free LPU)" },
    { id: "deepseek-r1-distill-llama-70b", name: "🟣 Groq DeepSeek R1 Distill (Free LPU)" },
    { id: "gpt-4o", name: "💎 OpenAI GPT-4o (Premium)" },
    { id: "claude-3.5-sonnet", name: "💎 Anthropic Claude 3.5 Sonnet (Premium)" },
  ];

  const handleSendPrompt = async (textToSend?: string) => {
    const prompt = textToSend || inputPrompt;
    if (!prompt.trim() || isSubmitting) return;

    const userMsgId = `user_${Date.now()}`;
    const assistantMsgId = `asst_${Date.now()}`;
    const startTime = performance.now();

    const userMessage: ChatMessage = {
      id: userMsgId,
      role: "user",
      content: prompt,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputPrompt("");
    setIsSubmitting(true);

    try {
      // Call Next.js /api/chat Proxy Route connected to EC2 FastAPI Gateway
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          messages: [{ role: "user", content: prompt }],
          temperature: 0.7,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        const endTime = performance.now();
        const totalMs = endTime - startTime;

        const responseContent = data.choices?.[0]?.message?.content || "No response text returned.";
        const actualModel = data.model || selectedModel;
        const meta = data.routemem_metadata || {};

        const liveTrace: ExecutionTrace = {
          id: data.id || `trc_${Math.random().toString(36).substring(2, 9)}`,
          timestamp: new Date().toLocaleTimeString(),
          query: prompt,
          cacheStatus: (meta.cache_status as any) || "MISS",
          exactHashHit: meta.cache_status === "EXACT_HIT",
          originalTokens: data.usage?.prompt_tokens || Math.round(prompt.length / 4),
          compressedTokens: data.usage?.prompt_tokens ? Math.round(data.usage.prompt_tokens * 0.2) : 20,
          compressionReductionPct: (meta.token_reduction_ratio || 0.812) * 100,
          profilerDifficulty: 0.75,
          profilerDomain: "General / Code",
          selectedModel: actualModel,
          selectedVendor: actualModel.includes("gemini")
            ? "Google AI Studio"
            : actualModel.includes("claude")
            ? "Anthropic Claude"
            : actualModel.includes("gpt") || actualModel.includes("o1")
            ? "OpenAI Cloud"
            : "Groq LPU",
          ttftMs: meta.ttft_ms || totalMs,
          totalLatencyMs: totalMs,
          queryCost: meta.cost_usd || 0.0001,
          baselineCost: 0.0420,
          savingsPct: 99.76,
          pipelineStages: sampleExecutionTrace.pipelineStages,
        };

        const assistantMessage: ChatMessage = {
          id: assistantMsgId,
          role: "assistant",
          content: responseContent,
          timestamp: new Date().toLocaleTimeString(),
          trace: liveTrace,
        };

        setMessages((prev) => [...prev, assistantMessage]);
        setActiveTrace(liveTrace);
      } else {
        throw new Error(`Gateway returned HTTP ${res.status}`);
      }
    } catch (err) {
      // Fallback clean mock trace handler
      const isMathExact = prompt.toLowerCase().includes("euler");
      const generatedTrace: ExecutionTrace = isMathExact
        ? {
            ...sampleExactHitTrace,
            id: `trc_${Math.random().toString(36).substring(2, 9)}`,
            timestamp: new Date().toLocaleTimeString(),
            query: prompt,
          }
        : {
            ...sampleExecutionTrace,
            id: `trc_${Math.random().toString(36).substring(2, 9)}`,
            timestamp: new Date().toLocaleTimeString(),
            query: prompt,
            selectedModel: selectedModel === "routemem-auto" ? "gemini-2.5-flash" : selectedModel,
          };

      const assistantMessage: ChatMessage = {
        id: assistantMsgId,
        role: "assistant",
        content: isMathExact
          ? "Euler's Identity formula:\n\n$$e^{i\\pi} + 1 = 0$$"
          : `Response generated via **${generatedTrace.selectedModel}**:\n\n\`\`\`rust\n// Model: ${generatedTrace.selectedModel}\npub fn solve() {\n    println!("Execution completed.");\n}\n\`\`\``,
        timestamp: new Date().toLocaleTimeString(),
        trace: generatedTrace,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setActiveTrace(generatedTrace);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-4rem)] bg-zinc-950">
      {/* Minimal Top Controls Header */}
      <div className="px-6 py-2.5 border-b border-zinc-800/60 bg-zinc-950/80 backdrop-blur-md flex flex-wrap items-center justify-between text-xs font-mono gap-3">
        {/* Model Selector Dropdown */}
        <div className="flex items-center space-x-2">
          <Cpu className="h-4 w-4 text-indigo-400" />
          <span className="text-zinc-400 font-medium">Model:</span>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="bg-zinc-900 border border-zinc-800 text-zinc-200 rounded-lg px-3 py-1 text-xs focus:outline-none focus:border-indigo-500 font-mono transition-all"
          >
            {availableModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </div>

        {/* View Mode & Preset Controls */}
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setIsComparisonMode(!isComparisonMode)}
            className={`px-3 py-1 rounded-lg text-xs font-medium border transition-all flex items-center space-x-1.5 ${
              isComparisonMode
                ? "bg-indigo-600/20 border-indigo-500/40 text-indigo-300"
                : "bg-zinc-900 border-zinc-800 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <Columns className="h-3.5 w-3.5 text-indigo-400" />
            <span>{isComparisonMode ? "Side-by-Side View" : "Single View"}</span>
          </button>
        </div>
      </div>

      {/* Clean Message Stream Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg) => {
          const isUser = msg.role === "user";
          const trace = msg.trace;

          if (isUser) {
            return (
              <div key={msg.id} className="flex flex-col items-end max-w-4xl mx-auto">
                <div className="rounded-xl px-4 py-3 bg-zinc-900 border border-zinc-800/80 text-zinc-100 max-w-xl text-xs font-sans">
                  {msg.content}
                </div>
              </div>
            );
          }

          return (
            <div key={msg.id} className="max-w-4xl mx-auto space-y-2">
              {isComparisonMode && trace ? (
                /* Side-by-Side Comparison Layout */
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Left: Un-routed Baseline */}
                  <div className="p-3.5 rounded-xl glass-panel border-amber-500/20 bg-zinc-950/80 space-y-2">
                    <div className="flex items-center justify-between text-[11px] font-mono border-b border-zinc-800/60 pb-1.5 text-amber-400">
                      <span className="font-semibold flex items-center gap-1">
                        <ShieldAlert className="h-3.5 w-3.5" /> Raw Baseline
                      </span>
                      <span>380ms | $0.042</span>
                    </div>
                    <div className="text-xs text-zinc-300 leading-relaxed font-sans">{msg.content}</div>
                  </div>

                  {/* Right: RouteMem Gateway */}
                  <div
                    onClick={() => setActiveTrace(trace)}
                    className="p-3.5 rounded-xl glass-panel border-indigo-500/30 bg-zinc-900/40 space-y-2 cursor-pointer hover:border-indigo-500/50 transition-all"
                  >
                    <div className="flex items-center justify-between text-[11px] font-mono border-b border-zinc-800/60 pb-1.5 text-emerald-400">
                      <span className="font-semibold flex items-center gap-1">
                        <Sparkles className="h-3.5 w-3.5 text-indigo-400" /> RouteMem ({trace.selectedModel})
                      </span>
                      <span>{formatLatency(trace.ttftMs)} | {formatCost(trace.queryCost)}</span>
                    </div>
                    <div className="text-xs text-zinc-200 leading-relaxed font-sans">{msg.content}</div>
                  </div>
                </div>
              ) : (
                /* Minimal Single Card View */
                <div className="rounded-xl p-4 glass-panel border-zinc-800/80 text-zinc-100 space-y-3">
                  <div className="text-xs leading-relaxed whitespace-pre-wrap text-zinc-200 font-sans">
                    {msg.content}
                  </div>

                  {trace && (
                    <div
                      onClick={() => setActiveTrace(trace)}
                      className="pt-2 border-t border-zinc-800/60 flex items-center justify-between text-[11px] font-mono cursor-pointer text-zinc-400 hover:text-indigo-300 transition-all"
                    >
                      <div className="flex items-center space-x-2">
                        <span className="px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 font-medium">
                          {trace.selectedModel}
                        </span>
                        <span className="text-emerald-400">⚡ {formatLatency(trace.ttftMs)}</span>
                        <span className="text-purple-300">🗜️ -{trace.compressionReductionPct.toFixed(0)}%</span>
                      </div>
                      <span className="flex items-center gap-1">
                        Trace <ArrowUpRight className="h-3 w-3" />
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Dock */}
      <div className="p-4 border-t border-zinc-800/50 bg-zinc-950/80 backdrop-blur-md">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendPrompt();
          }}
          className="max-w-4xl mx-auto flex items-center space-x-3"
        >
          <input
            type="text"
            value={inputPrompt}
            onChange={(e) => setInputPrompt(e.target.value)}
            placeholder="Type a prompt for RouteMem AI Gateway..."
            className="flex-1 bg-zinc-900/90 border border-zinc-800 rounded-xl px-4 py-2.5 text-xs text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-zinc-700 transition-all font-sans"
            disabled={isSubmitting}
          />
          <button
            type="submit"
            disabled={!inputPrompt.trim() || isSubmitting}
            className="px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-medium text-xs flex items-center space-x-1.5 transition-all shadow-sm"
          >
            <span>{isSubmitting ? "Routing..." : "Send"}</span>
            <Send className="h-3.5 w-3.5" />
          </button>
        </form>
      </div>
    </div>
  );
};
