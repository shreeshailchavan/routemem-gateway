import unittest
from fastapi.testclient import TestClient
from app.main import app

class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "RouteMem AI Gateway")

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(len(response.text) > 0)

    def test_chat_completions_endpoint(self):
        payload = {
            "model": "routemem-auto",
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"}
            ]
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("choices", data)
        self.assertIn("routemem_metadata", data)
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")

if __name__ == "__main__":
    unittest.main()
