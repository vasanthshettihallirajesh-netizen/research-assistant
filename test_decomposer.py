import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.decomposer import decompose, _strip_topic, _contains_trigger


class TestDecomposer(unittest.TestCase):
    def test_causal_question_gets_causal_templates(self):
        result = decompose("Does eating chocolate cause acne?")
        joined = " ".join(result).lower()
        self.assertIn("causal mechanism", joined)
        self.assertIn("controlled studies", joined)

    def test_comparison_question_gets_comparison_templates(self):
        result = decompose("Python vs JavaScript for backend development")
        joined = " ".join(result).lower()
        self.assertIn("key differences", joined)
        self.assertIn("comparison", joined)

    def test_no_false_positive_or_substring_in_more(self):
        """Regression test: 'or' should not match inside 'more'."""
        result = decompose("Is remote work more productive?")
        joined = " ".join(result).lower()
        # should get generic templates, not comparison templates
        self.assertIn("consensus", joined)
        self.assertNotIn("key differences", joined)

    def test_current_event_question_gets_current_templates(self):
        result = decompose("What is the latest news on the merger?")
        joined = " ".join(result).lower()
        self.assertIn("recent reporting", joined)

    def test_generic_fallback(self):
        result = decompose("Is climate change real?")
        self.assertGreater(len(result), 0)

    def test_strip_topic_removes_multiple_leading_words(self):
        self.assertEqual(_strip_topic("What is the latest news?"), "the latest news")
        self.assertEqual(_strip_topic("Does X cause Y?"), "X cause Y")

    def test_no_empty_or_duplicate_subquestions(self):
        result = decompose("Does remote work cause burnout?")
        self.assertEqual(len(result), len(set(result)), "should not contain duplicates")
        for sq in result:
            self.assertTrue(sq.strip(), "sub-question should not be empty")

    def test_contains_trigger_word_boundary(self):
        self.assertFalse(_contains_trigger("more productive", ["or"]))
        self.assertTrue(_contains_trigger("cats or dogs", ["or"]))


if __name__ == "__main__":
    unittest.main()
