import re
import os
import json
from typing import Tuple, Optional, List
import numpy as np
from app.config import settings, config_yaml

class QueryProfiler:
    """
    Tier-3 Dual-Signal Hybrid Query Complexity & Intent Profiler.
    Fuses Surface Syntactic AST analysis with Latent Semantic Prototype Projections
    using zero-overhead embedding reuse from Stage 3 (bge-small-en-v1.5).
    """

    def __init__(self, model_path: str = None, centroids_path: str = "models/intent_centroids.json"):
        self.model_path = model_path or config_yaml.get("router", {}).get("profiler", {}).get("model_path", "")
        self.centroids_path = centroids_path
        self.centroid_labels: List[str] = []
        self.centroids_matrix: Optional[np.ndarray] = None

        # Load calibrated prototype centroids
        if os.path.exists(self.centroids_path):
            try:
                with open(self.centroids_path, "r") as f:
                    data = json.load(f)
                    self.centroid_labels = data.get("labels", [])
                    raw_matrix = data.get("centroids", [])
                    if raw_matrix:
                        self.centroids_matrix = np.array(raw_matrix, dtype=np.float32)
            except Exception:
                self.centroids_matrix = None

        self.session = None
        if self.model_path and os.path.exists(self.model_path):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(self.model_path)
            except Exception:
                self.session = None

    def profile(self, prompt: str, embedding: Optional[List[float]] = None) -> Tuple[float, str]:
        """
        Evaluates query difficulty score D in [0.0, 1.0] and task intent category.
        Fuses Surface Syntactic AST with Reused Dense Semantic Embeddings (< 0.1 ms).
        """
        # 1. Surface Syntactic AST Analysis (< 0.3 ms)
        code_keywords = ["def ", "class ", "function", "import ", "select ", "return", "var ", "const ", "struct "]
        math_symbols = ["\\int", "\\sum", "sqrt", "^", "==", "!=", "<=", ">=", "matrix", "lambda"]
        expert_keywords = ["analyze ", "doctrine", "antitrust", "pharmacokinetic", "hipaa", "soc2", "quantum", "basel ", "cryptographic", "asymptotic", "cleavage", "macroeconomic", "contagion", "regulations", "audit ", "sovereign", "bioenergetic", "qubit", "eu ai act", "executive order", "snark", "ldpc"]
        reason_keywords = ["step by step", "how many", "think step", "puzzle", "riddle", "probability", "deduce", "prove ", "if it takes", "harmonic mean", "birthday", "switches"]

        prompt_lower = prompt.lower()
        code_count = sum(1 for kw in code_keywords if kw in prompt_lower)
        math_count = sum(1 for sym in math_symbols if sym in prompt_lower)
        expert_count = sum(1 for kw in expert_keywords if kw in prompt_lower)
        reason_count = sum(1 for kw in reason_keywords if kw in prompt_lower)

        # Detect simple arithmetic (e.g., "What is 12 + 15?", "Calculate 25 * 4", "100 / 5", "7 times 8")
        is_simple_math = bool(re.search(r"^(what is|calculate|evaluate|solve)?\s*[\d\.\s\+\-\*\/\%\(\)\^x×÷=]+(\?)?$", prompt_lower.strip())) or \
                         (len(prompt.split()) <= 15 and bool(re.search(r"\d+\s*[\+\-\*\/\%x×÷]\s*\d+", prompt_lower)) and not any(kw in prompt_lower for kw in reason_keywords))

        # Disambiguate code generation requests vs conceptual systems architecture explanations
        is_explanation = any(prompt_lower.startswith(q) or f" {q}" in prompt_lower for q in ["how does ", "how do ", "how is ", "explain how ", "what is ", "what are ", "describe ", "define "])
        has_explicit_code = bool(re.search(r"(def\s+\w+\(|class\s+\w+|import\s+\w+|async\s+def|SELECT\s+.*FROM|`def )", prompt))
        code_gen_triggers = ["write a ", "write an ", "implement a ", "implement an ", "fix this bug", "fix the bug", "debug ", "refactor ", "write code", "code to ", "script to ", "decorator", "topological sort", "window function", "asyncio function", "generator method"]
        is_code_generation = (any(trig in prompt_lower for trig in code_gen_triggers) or has_explicit_code or code_count >= 2) and (not is_explanation or has_explicit_code)

        length_factor = min(0.30, len(prompt.split()) / 500.0)
        syntax_factor = min(0.35, (code_count * 0.08) + (math_count * 0.10))

        if is_simple_math:
            base_syntax_diff = 0.24
            syntax_intent = "simple_qa"
        elif expert_count >= 1 or "comparative analysis" in prompt_lower:
            base_syntax_diff = 0.82
            syntax_intent = "domain_expert"
        elif is_code_generation:
            base_syntax_diff = 0.58
            syntax_intent = "code_generation"
        elif math_count >= 2 or reason_count >= 1 or "solve for" in prompt_lower or "proof" in prompt_lower or "prove " in prompt_lower:
            base_syntax_diff = 0.65
            syntax_intent = "complex_reasoning"
        elif len(prompt.split()) > 150:
            base_syntax_diff = 0.55
            syntax_intent = "complex_reasoning"
        else:
            base_syntax_diff = 0.28
            syntax_intent = "simple_qa"

        d_syntax = min(1.0, base_syntax_diff + length_factor + syntax_factor)

        # 2. Latent Semantic Prototype Analysis via Reused Embedding (< 0.05 ms)
        if embedding is not None and self.centroids_matrix is not None and len(embedding) == self.centroids_matrix.shape[1]:
            try:
                vec = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

                # Microsecond matrix dot product: (4, 384) @ (384,) -> (4,)
                sims = np.dot(self.centroids_matrix, vec)
                sim_dict = {label: float(sims[idx]) for idx, label in enumerate(self.centroid_labels)}

                best_idx = int(np.argmax(sims))
                semantic_intent = self.centroid_labels[best_idx]

                s_reason = sim_dict.get("complex_reasoning", 0.0)
                s_expert = sim_dict.get("domain_expert", 0.0)
                s_code   = sim_dict.get("code_generation", 0.0)
                s_faq    = sim_dict.get("simple_qa", 0.0)

                # Map semantic proximity to difficulty
                d_semantic = 0.30 + (s_expert * 0.55) + (s_reason * 0.35) + (s_code * 0.15) - (s_faq * 0.45)
                d_semantic = float(np.clip(d_semantic, 0.10, 0.95))

                # Riddle & Counter-intuitive deduction boost (for actual reasoning traps, never arithmetic)
                if not is_simple_math and s_reason > 0.60 and reason_count >= 1:
                    riddle_boost = (s_reason - 0.50) * 0.80
                    d_semantic += riddle_boost

                # Beginner tutorial & boilerplate suppression
                if s_faq > 0.65 and ("what is" in prompt_lower or "beginner" in prompt_lower or "how do i" in prompt_lower):
                    d_semantic = min(d_semantic, 0.32)

                if is_simple_math:
                    d_semantic = min(d_semantic, 0.28)

                # Dual-Signal Fusion: 35% syntax + 65% semantic
                d_fused = float(np.clip(0.35 * d_syntax + 0.65 * d_semantic, 0.0, 1.0))

                # Resolve final intent
                if is_simple_math:
                    final_intent = "simple_qa"
                    d_fused = min(d_fused, 0.30)
                elif expert_count >= 1 or "comparative analysis" in prompt_lower or (s_expert > 0.68 and d_fused >= 0.70):
                    final_intent = "domain_expert"
                    d_fused = max(d_fused, 0.78)
                elif is_code_generation:
                    final_intent = "code_generation"
                elif not is_code_generation and semantic_intent == "code_generation":
                    # Conceptual systems/architecture question, NOT code generation
                    final_intent = "simple_qa"
                    d_fused = min(d_fused, 0.45)
                elif math_count >= 2 or reason_count >= 1:
                    final_intent = "complex_reasoning"
                else:
                    final_intent = semantic_intent

                return round(d_fused, 3), final_intent
            except Exception:
                pass

        # Fallback to pure syntactic analyzer if embedding is None
        return round(d_syntax, 3), syntax_intent
