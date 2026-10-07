import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from app.config import settings, models_yaml
from app.router.uniroute import UniRouteMapper
from app.utils.logger import get_logger

logger = get_logger("omnirouter")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ONNX_MODEL_PATH = BASE_DIR / "models" / "preference_head.onnx"

class OmniRouter:
    """OmniRouter Lagrangian Dual Budget Solver with RouteLLM ONNX Neural Preference Head."""

    def __init__(
        self,
        alpha_target: float = 0.95,
        monthly_budget_cap: float = 500.0,
        lambda_quality: float = 1.0
    ):
        self.alpha_target = alpha_target
        self.monthly_budget_cap = monthly_budget_cap
        self.lambda_quality = lambda_quality
        self.mapper = UniRouteMapper()
        self.models_spec = models_yaml.get("models", {
            "llama-3.2-3b-local": {"cost_per_token": 0.0, "base_accuracy": 0.89},
            "qwen2.5-coder:3b": {"cost_per_token": 0.0, "base_accuracy": 0.93},
            "deepseek-r1:1.5b": {"cost_per_token": 0.0, "base_accuracy": 0.94},
            "llama-3.1-8b": {"cost_per_token": 0.000002, "base_accuracy": 0.82},
            "qwen-2.5-coder-32b": {"cost_per_token": 0.000015, "base_accuracy": 0.91},
            "claude-3.5-sonnet": {"cost_per_token": 0.000350, "base_accuracy": 0.98}
        })

        # Load RouteLLM ONNX Neural Preference Head (<0.1ms execution)
        self.ort_session = None
        if ONNX_MODEL_PATH.exists():
            try:
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
                logger.info(f"Loaded RouteLLM ONNX Neural Preference Head from {ONNX_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Failed to initialize ONNX Runtime session for RouteLLM head: {e}")

    def predict_slm_win_probability(
        self,
        difficulty: float,
        intent: str,
        code_density: float = 0.0,
        math_density: float = 0.0,
        length_norm: float = 0.5
    ) -> float:
        """Evaluates RouteLLM pairwise preference head predicting P(Local SLM satisfies query >= Cloud)."""
        if self.ort_session is None:
            # Analytical baseline heuristic
            return float(np.clip(1.0 - (difficulty * 0.70) - (code_density * 0.15) - (math_density * 0.15), 0.0, 1.0))

        intent_vec = [0.0, 0.0, 0.0, 0.0]
        i_lower = intent.lower()
        if "code" in i_lower:
            intent_vec[0] = 1.0
        elif "math" in i_lower:
            intent_vec[1] = 1.0
        elif "reason" in i_lower:
            intent_vec[2] = 1.0
        else:
            intent_vec[3] = 1.0

        local_cap = [0.89, 0.82, 0.85, 0.96]
        cloud_cap = [0.98, 0.97, 0.96, 0.60]

        features = np.array([[
            difficulty, code_density, math_density, length_norm,
            intent_vec[0], intent_vec[1], intent_vec[2], intent_vec[3],
            local_cap[0], local_cap[1], local_cap[2], local_cap[3],
            cloud_cap[0], cloud_cap[1], cloud_cap[2], cloud_cap[3]
        ]], dtype=np.float32)

        try:
            outputs = self.ort_session.run(None, {"input": features})
            return float(outputs[0][0][0])
        except Exception as e:
            logger.error(f"Error during ONNX preference evaluation: {e}")
            return float(np.clip(1.0 - (difficulty * 0.70), 0.0, 1.0))

    def select_model(
        self,
        difficulty: float,
        intent: str,
        max_cost_target: Optional[float] = None,
        quality_target: Optional[float] = None,
        code_density: float = 0.0,
        math_density: float = 0.0
    ) -> str:
        """
        Routes query using RouteLLM Neural Preference Head combined with Lagrangian Dual Budget Optimization.
        """
        # 1. Neural Pairwise Preference Evaluation (<0.1ms)
        slm_win_prob = self.predict_slm_win_probability(
            difficulty=difficulty,
            intent=intent,
            code_density=code_density,
            math_density=math_density
        )

        # 2. Fast-Path Local SLM Resolution if Neural Head predicts local model satisfies query
        if slm_win_prob >= 0.50:
            i_lower = intent.lower()
            if "code" in i_lower:
                return "qwen2.5-coder:3b"
            elif "math" in i_lower or "reason" in i_lower:
                return "deepseek-r1:1.5b"
            elif difficulty <= 0.25:
                return "phi3.5:latest"
            else:
                return "llama-3.2-3b-local"

        # 3. Escalation: Solve Lagrangian Dual Optimization for Cloud Frontier Fleet
        target_quality = quality_target or self.alpha_target
        target_vec = self.mapper.map_intent_to_target_vector(intent, difficulty)
        candidate_models = self.mapper.find_capable_models(target_vec)

        best_model = None
        min_lagrangian_score = float("inf")

        for model_id in candidate_models:
            specs = self.models_spec.get(model_id, {})
            cost = specs.get("cost_per_token", 0.00005)
            base_acc = specs.get("base_accuracy", 0.85)

            # Adjust predicted accuracy based on query difficulty
            predicted_acc = base_acc * (1.0 - 0.2 * max(0.0, difficulty - 0.5))

            if max_cost_target is not None and cost > max_cost_target:
                continue

            lagrangian_score = cost * 1000.0 - self.lambda_quality * (predicted_acc - target_quality)

            if lagrangian_score < min_lagrangian_score:
                min_lagrangian_score = lagrangian_score
                best_model = model_id

        if not best_model:
            best_model = "openai/gpt-oss-120b"

        return best_model
