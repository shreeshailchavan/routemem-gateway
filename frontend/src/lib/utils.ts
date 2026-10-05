import { ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCost(amount: number): string {
  if (amount === 0) return "$0.0000";
  if (amount < 0.0001) return `< $0.0001`;
  return `$${amount.toFixed(4)}`;
}

export function formatLatency(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)} μs`;
  return `${ms.toFixed(2)} ms`;
}
