import unittest
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db


class TestDB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        db.DB_PATH = self.tmp.name
        db.init_db()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_create_and_get_topic(self):
        tid = db.create_topic("Does X cause Y?")
        topic = db.get_topic(tid)
        self.assertEqual(topic["question"], "Does X cause Y?")
        self.assertEqual(topic["status"], "open")
        self.assertEqual(topic["sub_questions"], [])

    def test_get_nonexistent_topic_returns_none(self):
        self.assertIsNone(db.get_topic(9999))

    def test_add_sub_questions_and_answer(self):
        tid = db.create_topic("Test question?")
        ids = db.add_sub_questions(tid, ["sub q 1", "sub q 2"])
        self.assertEqual(len(ids), 2)

        db.answer_sub_question(ids[0], "the answer", 0.9)
        topic = db.get_topic(tid)
        answered = [sq for sq in topic["sub_questions"] if sq["status"] == "answered"]
        self.assertEqual(len(answered), 1)
        self.assertEqual(answered[0]["answer"], "the answer")
        self.assertEqual(answered[0]["confidence"], 0.9)

    def test_mark_unresolved(self):
        tid = db.create_topic("Test?")
        ids = db.add_sub_questions(tid, ["sub q"])
        db.mark_sub_question_unresolved(ids[0], "no data available")
        topic = db.get_topic(tid)
        self.assertEqual(topic["sub_questions"][0]["status"], "unresolved")
        self.assertEqual(topic["sub_questions"][0]["answer"], "no data available")

    def test_add_source_attaches_to_subquestion(self):
        tid = db.create_topic("Test?")
        ids = db.add_sub_questions(tid, ["sub q"])
        db.add_source(ids[0], "https://example.com", "Title", "excerpt text", "supports")
        topic = db.get_topic(tid)
        self.assertEqual(len(topic["sub_questions"][0]["sources"]), 1)
        self.assertEqual(topic["sub_questions"][0]["sources"][0]["stance"], "supports")

    def test_update_topic_status(self):
        tid = db.create_topic("Test?")
        db.update_topic_status(tid, "resolved", "Final summary here")
        topic = db.get_topic(tid)
        self.assertEqual(topic["status"], "resolved")
        self.assertEqual(topic["summary"], "Final summary here")

    def test_find_similar_topics_semantic_match(self):
        db.create_topic("Does remote work reduce productivity?")
        db.create_topic("Best pizza toppings")
        results = db.find_similar_topics("productivity effects of working from home")
        matched_questions = [r["question"] for r in results]
        self.assertIn("Does remote work reduce productivity?", matched_questions)
        self.assertNotIn("Best pizza toppings", matched_questions)

    def test_get_stats_counts_correctly(self):
        tid = db.create_topic("Test?")
        db.add_sub_questions(tid, ["a", "b"])
        db.update_topic_status(tid, "resolved")
        stats = db.get_stats()
        self.assertEqual(stats["total_topics"], 1)
        self.assertEqual(stats["resolved_topics"], 1)
        self.assertEqual(stats["total_sub_questions"], 2)


if __name__ == "__main__":
    unittest.main()
