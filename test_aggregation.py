import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.aggregation import assess_topic


class TestAggregation(unittest.TestCase):
    def test_no_subquestions_returns_zero_confidence(self):
        result = assess_topic({"sub_questions": []})
        self.assertEqual(result.overall_confidence, 0.0)
        self.assertEqual(result.resolved_ratio, 0.0)

    def test_all_answered_no_contradiction(self):
        topic = {
            "sub_questions": [
                {"id": 1, "status": "answered", "confidence": 0.8, "sources": [{"stance": "supports"}]},
                {"id": 2, "status": "answered", "confidence": 0.6, "sources": [{"stance": "supports"}]},
            ]
        }
        result = assess_topic(topic)
        self.assertFalse(result.has_contradictions)
        self.assertAlmostEqual(result.overall_confidence, 0.7, places=5)
        self.assertEqual(result.resolved_ratio, 1.0)

    def test_contradiction_detected_and_discounted(self):
        topic = {
            "sub_questions": [
                {
                    "id": 1, "status": "answered", "confidence": 0.8,
                    "sources": [{"stance": "supports"}, {"stance": "contradicts"}],
                },
            ]
        }
        result = assess_topic(topic)
        self.assertTrue(result.has_contradictions)
        self.assertIn(1, result.contradicted_sub_questions)
        self.assertAlmostEqual(result.overall_confidence, 0.4, places=5)  # 0.8 * 0.5

    def test_neutral_and_supports_not_flagged_as_contradiction(self):
        topic = {
            "sub_questions": [
                {
                    "id": 1, "status": "answered", "confidence": 0.7,
                    "sources": [{"stance": "supports"}, {"stance": "neutral"}],
                },
            ]
        }
        result = assess_topic(topic)
        self.assertFalse(result.has_contradictions)

    def test_unanswered_subquestions_reduce_resolved_ratio(self):
        topic = {
            "sub_questions": [
                {"id": 1, "status": "answered", "confidence": 0.5, "sources": []},
                {"id": 2, "status": "open", "confidence": None, "sources": []},
            ]
        }
        result = assess_topic(topic)
        self.assertEqual(result.resolved_ratio, 0.5)

    def test_unresolved_generates_note(self):
        topic = {
            "sub_questions": [
                {"id": 1, "status": "unresolved", "confidence": None, "sources": [], "answer": "no data found"},
            ]
        }
        result = assess_topic(topic)
        self.assertTrue(any("unresolved" in n.lower() for n in result.notes))


if __name__ == "__main__":
    unittest.main()
