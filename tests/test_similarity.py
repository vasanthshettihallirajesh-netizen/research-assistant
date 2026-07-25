import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.similarity import tokenize, compute_tf, compute_idf, cosine_similarity, rank_by_similarity


class TestSimilarity(unittest.TestCase):
    def test_tokenize_removes_stopwords(self):
        tokens = tokenize("What is the current state of the economy?")
        self.assertNotIn("what", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("the", tokens)
        self.assertIn("current", tokens)
        self.assertIn("economy", tokens)

    def test_identical_text_has_similarity_1(self):
        text = "remote work productivity study"
        tf = compute_tf(tokenize(text))
        idf = compute_idf([tokenize(text)])
        vec = {t: f * idf.get(t, 0) for t, f in tf.items()}
        score = cosine_similarity(vec, vec)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_unrelated_text_has_low_similarity(self):
        results = rank_by_similarity(
            "remote work and productivity",
            ["Does chocolate cause acne?", "Best pizza toppings"],
        )
        # both unrelated, expect empty or very low-scored results
        for score, _ in results:
            self.assertLess(score, 0.1)

    def test_reworded_question_still_matches(self):
        results = rank_by_similarity(
            "remote work and productivity levels",
            [
                "Does remote work reduce productivity?",
                "What are the productivity effects of working from home?",
                "Best pizza toppings",
            ],
        )
        matched_texts = [text for score, text in results]
        self.assertIn("Does remote work reduce productivity?", matched_texts)

    def test_ranking_order_makes_sense(self):
        results = rank_by_similarity(
            "remote work productivity",
            [
                "Does remote work reduce productivity?",  # near-exact match
                "Best pizza toppings",                      # unrelated
            ],
        )
        self.assertEqual(results[0][1], "Does remote work reduce productivity?")

    def test_empty_query_returns_no_crash(self):
        results = rank_by_similarity("", ["some topic"])
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
