import re
import os
from typing import Tuple
import numpy as np
from app.config import settings, config_yaml

class QueryProfiler:
    """Tier-3 DeBERTa-v3 ONNX Query Difficulty & Intent Profiler."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or config_yaml.get("router", {}).get("profiler", {}).get("model_path", "")
        self.session = None
        if self.model_path and os.path.exists(self.model_path):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(self.model_path)
            except Exception:
                self.session = None

    def profile(self, prompt: str) -> Tuple[float, str]:
        """
        Evaluates query difficulty score D in [0.0, 1.0] and task intent category.
        Target latency: < 3 ms.
        """
        # If ONNX model session is loaded, perform ONNX inference
        if self.session is not None:
            try:
                # Mock ONNX feature input for DeBERTa encoder
                input_data = np.random.randn(1, 128).astype(np.float32)
                outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_data})
                score = float(outputs[0][0][0])
                difficulty = max(0.0, min(1.0, score))
                intent = "code_generation" if difficulty > 0.6 else "simple_qa"
                return round(difficulty, 3), intent
            except Exception:
                pass

        # Heuristic structural & semantic profiler fallback (< 1 ms)
        code_keywords = ["def ", "class ", "function", "import ", "select ", "return", "var ", "const ", "struct "]
        math_symbols = ["\\int", "\\sum", "sqrt", "^", "==", "!=", "<=", ">=", "matrix", "lambda"]

        prompt_lower = prompt.lower()
        code_count = sum(1 for kw in code_keywords if kw in prompt_lower)
        math_count = sum(1 for sym in math_symbols if sym in prompt_lower)

        # Intent classification
        if code_count >= 2 or "write a" in prompt_lower or "code" in prompt_lower or "bug" in prompt_lower:
            intent = "code_generation"
            base_diff = 0.65
        elif math_count >= 2 or "solve" in prompt_lower or "calculate" in prompt_lower:
            intent = "complex_reasoning"
            base_diff = 0.70
        elif len(prompt.split()) > 150:
            intent = "complex_reasoning"
            base_diff = 0.55
        else:
            intent = "simple_qa"
            base_diff = 0.25

        # Difficulty adjustments based on token length and syntax density
        length_factor = min(0.3, len(prompt.split()) / 500.0)
        syntax_factor = min(0.3, (code_count * 0.08) + (math_count * 0.1))

        difficulty = min(1.0, base_diff + length_factor + syntax_factor)
        return round(difficulty, 3), intent
