from typing import Dict, List, Any
import numpy as np
from app.config import models_yaml

class UniRouteMapper:
    """UniRoute continuous capability vector space mapper."""

    def __init__(self):
        self.models_spec = models_yaml.get("models", {})

    def map_intent_to_target_vector(self, intent: str, difficulty: float) -> np.ndarray:
        """
        Maps (intent, difficulty) to target capability vector [reasoning, code, math, speed].
        """
        if intent == "code_generation":
            target = np.array([0.70, 0.90, 0.60, 0.70])
        elif intent == "complex_reasoning":
            target = np.array([0.95, 0.70, 0.85, 0.50])
        else:  # simple_qa
            target = np.array([0.50, 0.30, 0.40, 0.95])

        target = target * (0.6 + 0.4 * difficulty)
        return target

    def find_capable_models(self, target_vector: np.ndarray) -> List[str]:
        """Returns candidate models matching or exceeding capability threshold."""
        capable = []
        for model_id, specs in self.models_spec.items():
            cap_vec = np.array(specs.get("capability_vector", [0.5, 0.5, 0.5, 0.5]))
            similarity = np.dot(cap_vec, target_vector) / (np.linalg.norm(cap_vec) * np.linalg.norm(target_vector) + 1e-6)
            if similarity > 0.65:
                capable.append(model_id)

        return capable if capable else list(self.models_spec.keys())
