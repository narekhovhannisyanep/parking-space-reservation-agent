"""Offline checks for RAG evaluation metrics and the harness (no provider calls)."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from test_static_storage import KeywordEmbeddings

from rag_evaluation import (
    ANSWER_CASES,
    RETRIEVAL_CASES,
    AnswerCase,
    Evaluator,
    capped_recall_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from reservation_collection import ReservationExtraction
from storage.seed import load_spaces


class MetricTests(unittest.TestCase):
    def test_rank_metrics(self):
        relevant = frozenset({"A", "B"})
        self.assertEqual(recall_at_k(["A", "X", "B", "C"], relevant, k=3), 1.0)
        self.assertEqual(recall_at_k(["X", "Y", "Z", "A"], relevant, k=3), 0.0)
        many = frozenset({"A", "B", "C", "D"})
        self.assertEqual(recall_at_k(["A", "B", "C"], many, k=3), 0.75)
        self.assertEqual(capped_recall_at_k(["A", "B", "C"], many, k=3), 1.0)
        self.assertAlmostEqual(precision_at_k(["A", "X"], relevant, k=3), 1 / 3)
        self.assertEqual(reciprocal_rank(["X", "B"], relevant), 0.5)
        self.assertEqual(reciprocal_rank([], relevant), 0.0)
        self.assertEqual(percentile([1, 2, 3, 4], 0.95), 4)
        self.assertEqual(percentile([], 0.5), 0.0)

    def test_labels_reference_fixture_spaces(self):
        inventory = {space.space_id for space in load_spaces()}
        for case in RETRIEVAL_CASES:
            with self.subTest(case=case.identifier):
                self.assertTrue(case.relevant)
                self.assertLessEqual(case.relevant, inventory)
        self.assertEqual(len({case.identifier for case in RETRIEVAL_CASES + ANSWER_CASES}), 33)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.reply = "No information."
        self.evaluator = Evaluator(
            Path(directory.name),
            embeddings=KeywordEmbeddings(),
            llm=GenericFakeChatModel(messages=iter(lambda: AIMessage(self.reply), None)),
            extractor=RunnableLambda(lambda _: {"parsed": ReservationExtraction(), "parsing_error": None}),
        )
        self.evaluator.static.initialize()

    def test_retrieval_report_shape_and_invariants(self):
        report = self.evaluator.evaluate_retrieval()
        self.assertEqual(report["cases"], len(RETRIEVAL_CASES))
        for row in report["per_case"]:
            self.assertLessEqual(len(row["offered"]), 3)
        self.assertEqual(report["follow_ups"]["failures"], [])
        self.assertIn("context_augmentation", report["latency"])

    def test_answer_checks_are_rule_based(self):
        case = AnswerCase("x", ("q",), ("no_booking_claim", "mentions_offered", "no_unoffered_ids"))
        check = self.evaluator.check
        self.assertFalse(check("no_booking_claim", case, "Space G-E02 is now reserved for you.", [], []))
        self.assertTrue(check("no_booking_claim", case, "These are suggestions only.", [], []))
        self.assertTrue(check("mentions_offered", case, "Try E02 or G-A01.", ["G-E02", "G-A01"], []))
        self.assertFalse(check("no_unoffered_ids", case, "Try G-A02.", ["G-E02"], []))
        self.assertTrue(check("offers_disjoint", case, "", ["G-A01"], ["G-E02"]))
        for reply in ("I don’t have the opening hours.", "Prices aren’t available.", "Hours are unknown."):
            self.assertTrue(check("states_unknown", case, reply, None, []))
        self.assertFalse(check("states_unknown", case, "We open at 08:00.", None, []))

    def test_answer_report_counts_failures_and_cleans_facts(self):
        report = self.evaluator.evaluate_answers()
        self.assertEqual(report["cases"], len(ANSWER_CASES))
        failed = {row["case"] for row in report["failures"]}
        self.assertIn("hours-known", failed)
        self.assertNotIn("hours-missing", failed)
        self.assertEqual(self.evaluator.dynamic.list_facts("demo-central"), [])


if __name__ == "__main__":
    unittest.main()
