import unittest
from app.router.profiler import QueryProfiler
from app.router.omnirouter import OmniRouter

class TestRouter(unittest.TestCase):

    def test_query_profiler_difficulty_and_intent(self):
        profiler = QueryProfiler()

        simple_query = "What is the capital of France?"
        diff_simple, intent_simple = profiler.profile(simple_query)
        self.assertEqual(intent_simple, "simple_qa")
        self.assertLess(diff_simple, 0.50)

        code_query = "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]"
        diff_code, intent_code = profiler.profile(code_query)
        self.assertEqual(intent_code, "code_generation")
        self.assertGreater(diff_code, 0.50)

    def test_omnirouter_model_selection(self):
        router = OmniRouter(alpha_target=0.95)

        # Simple QA with max cost cap routes to lowest cost local SLM or free API
        selected_simple = router.select_model(difficulty=0.2, intent="simple_qa", max_cost_target=0.000005)
        self.assertIn(selected_simple, ["llama-3.1-8b", "groq-gpt-120b", "groq-qwen-27b", "gemini-3.8-flash", "openrouter-free-deepseek-r1"])

        # High difficulty code query routes to capable code or frontier model
        selected_code = router.select_model(difficulty=0.85, intent="code_generation")
        self.assertIn(selected_code, ["qwen-2.5-coder-32b", "claude-3.5-sonnet", "openrouter-free-deepseek-r1", "deepseek-v3", "deepseek-r1", "groq-gpt-120b", "groq-qwen-27b"])

if __name__ == "__main__":
    unittest.main()
