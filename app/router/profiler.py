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

        prompt_lower = prompt.lower()
        code_count = sum(1 for kw in code_keywords if kw in prompt_lower)
        math_count = sum(1 for sym in math_symbols if sym in prompt_lower)

        length_factor = min(0.30, len(prompt.split()) / 500.0)
        syntax_factor = min(0.35, (code_count * 0.08) + (math_count * 0.10))

        if code_count >= 2 or "write a" in prompt_lower or "code" in prompt_lower or "bug" in prompt_lower:
            base_syntax_diff = 0.65
            syntax_intent = "code_generation"
        elif math_count >= 2 or "solve" in prompt_lower or "calculate" in prompt_lower:
            base_syntax_diff = 0.70
            syntax_intent = "complex_reasoning"
        elif len(prompt.split()) > 150:
            base_syntax_diff = 0.55
            syntax_intent = "complex_reasoning"
        else:
            base_syntax_diff = 0.25
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
                d_semantic = 0.50 + (s_reason * 0.45) + (s_expert * 0.40) + (s_code * 0.30) - (s_faq * 0.45)
                d_semantic = float(np.clip(d_semantic, 0.10, 0.95))

                # Riddle & Counter-intuitive deduction boost (fixes "Sally's brothers" trap)
                if s_reason > 0.60 and d_syntax < 0.45:
                    riddle_boost = (s_reason - 0.50) * 0.80
                    d_semantic += riddle_boost

                # Beginner tutorial & boilerplate suppression (fixes "What is def" trap)
                if s_faq > 0.65 and ("what is" in prompt_lower or "beginner" in prompt_lower or "how do i" in prompt_lower):
                    d_semantic = min(d_semantic, 0.32)

                # Dual-Signal Fusion: 35% syntax + 65% semantic
                d_fused = float(np.clip(0.35 * d_syntax + 0.65 * d_semantic, 0.0, 1.0))
                
                # Math override if explicit math symbols present
                final_intent = "complex_reasoning" if math_count >= 2 else semantic_intent
                return round(d_fused, 3), final_intent
            except Exception:
                pass

        # Fallback to pure syntactic analyzer if embedding is None
        return round(d_syntax, 3), syntax_intent
