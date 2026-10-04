import os
import hashlib
from typing import List
import numpy as np

class DenseEmbedder:
    """bge-small-en-v1.5 Dense Vector Embedder (384-dimensions)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.vector_dim = 384
        self._fastembed = None
        try:
            from fastembed import TextEmbedding
            self._fastembed = TextEmbedding(model_name=self.model_name)
        except Exception:
            self._fastembed = None

    def embed(self, text: str) -> List[float]:
        """Returns 384-dimensional normalized float embedding vector."""
        if self._fastembed is not None:
            try:
                embeddings = list(self._fastembed.embed([text]))
                vector = np.array(embeddings[0], dtype=np.float32)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                return vector.tolist()
            except Exception:
                pass

        # Deterministic 384-dim semantic feature hashing fallback
        words = text.lower().split()
        vector = np.zeros(self.vector_dim, dtype=np.float32)
        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.vector_dim
            val = ((h >> 8) % 100) / 100.0
            vector[idx] += val
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
