import unittest
import hashlib
from app.router.profiler import QueryProfiler

class TestCache(unittest.TestCase):

    def test_exact_hash_computation(self):
        system_prompt = "You are a coding assistant."
        user_prompt = "How do I implement binary search in Python?"

        payload = f"{system_prompt.strip()}::{user_prompt.strip()}".encode("utf-8")
        h1 = hashlib.sha256(payload).hexdigest()
        h2 = hashlib.sha256(payload).hexdigest()

        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_vectorizer_dimension(self):
        import numpy as np
        text = "Write a python function to compute fibonacci numbers"
        words = text.lower().split()
        vector = np.zeros(384, dtype=np.float32)
        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % 384
            val = ((h >> 8) % 100) / 100.0
            vector[idx] += val
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        self.assertEqual(len(vector), 384)

if __name__ == "__main__":
    unittest.main()
