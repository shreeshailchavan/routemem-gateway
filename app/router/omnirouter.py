from typing import Optional, Dict, Any
from app.config import settings, models_yaml
from app.router.uniroute import UniRouteMapper

class OmniRouter:
    """OmniRouter Lagrangian Dual Budget Solver."""

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
            "llama-3.1-8b": {"cost_per_token": 0.000002, "base_accuracy": 0.82},
            "qwen-2.5-coder-32b": {"cost_per_token": 0.000015, "base_accuracy": 0.91},
            "claude-3.5-sonnet": {"cost_per_token": 0.000350, "base_accuracy": 0.98}
        })

    def select_model(
        self,
        difficulty: float,
        intent: str,
        max_cost_target: Optional[float] = None,
        quality_target: Optional[float] = None
    ) -> str:
        """
        Solves Lagrangian dual optimization problem:
        Loss = Cost - lambda * (Predicted_Accuracy - Quality_Target)
        """
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

            lagrangian_score = cost - self.lambda_quality * (predicted_acc - target_quality)

            if lagrangian_score < min_lagrangian_score:
                min_lagrangian_score = lagrangian_score
                best_model = model_id

        if not best_model:
            best_model = "llama-3.1-8b"

        return best_model
