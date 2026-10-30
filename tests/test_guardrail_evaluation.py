"""Verify evaluation counts do not reward partial PII protection."""

import unittest

from guardrail_evaluation import Case, evaluate
from guardrails import PrivacyGuard, SensitiveSpan


class EvaluationTests(unittest.TestCase):
    def test_full_partial_and_false_positive_cases_are_counted(self):
        guard = PrivacyGuard(lambda text: [SensitiveSpan(0, 2, "PERSON")] if text in {"AB", "ABC", "XY"} else [])
        report = evaluate(
            guard,
            (
                Case("full", "AB", ("AB",)),
                Case("partial", "ABC", ("ABC",)),
                Case("false-positive", "XY"),
                Case("clean", "clean"),
            ),
        )
        self.assertEqual(report["counts"], {"tp": 1, "fp": 1, "fn": 1, "tn": 1})
        self.assertEqual(report["precision"], 0.5)
        self.assertEqual(report["recall"], 0.5)
        self.assertNotIn("ABC", str(report))

    def test_all_sensitive_values_and_repeated_occurrences_require_coverage(self):
        guard = PrivacyGuard(lambda text: [SensitiveSpan(0, 2, "PERSON")] if text.startswith("AB") else [])
        report = evaluate(
            guard,
            (
                Case("repeated", "AB AB", ("AB",)),
                Case("mixed", "AB CD", ("AB", "CD")),
            ),
        )
        self.assertEqual(report["counts"]["fn"], 2)
        self.assertEqual(report["counts"]["tp"], 0)


if __name__ == "__main__":
    unittest.main()
