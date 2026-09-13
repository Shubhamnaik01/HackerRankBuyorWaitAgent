from __future__ import annotations

import unittest
from decimal import Decimal

from code.evaluation.usage_report import estimate_cost, render_usage_report
from code.evidence.usage import UsageTracker


class UsageReportTests(unittest.TestCase):
    def test_zero_call_tracker(self):
        report = render_usage_report(UsageTracker(), 250)
        self.assertIn("Total model calls: 0", report)
        self.assertIn("Estimated total API cost (USD): 0.00000000", report)

    def test_one_model_aggregation_and_cost(self):
        usage = UsageTracker()
        usage.record("gpt-5-mini", "message", 1000, 500, 1500)
        report = render_usage_report(usage, 10)
        self.assertIn("| openai | gpt-5-mini | 1 | 1000 | 500 | 1500 | 0.00125000 |", report)
        self.assertEqual(Decimal("0.00125"), estimate_cost("openai", "gpt-5-mini", 1000, 500))

    def test_multiple_purposes_and_models(self):
        usage = UsageTracker()
        usage.record("gpt-5-mini", "message", 100, 10, 110)
        usage.record("gpt-5-mini", "image", 200, 20, 220)
        usage.record("gpt-5-mini-2025-08-07", "image", 300, 30, 330)
        report = render_usage_report(usage, 3)
        self.assertIn("Message extraction calls: 1", report)
        self.assertIn("Image extraction calls: 2", report)
        self.assertIn("gpt-5-mini-2025-08-07", report)
        self.assertIn("Total model calls: 3", report)

    def test_cache_hits_and_validation_failures_do_not_add_usage(self):
        usage = UsageTracker()
        usage.record_cache_hit()
        usage.record_validation_failure()
        report = render_usage_report(usage, 1)
        self.assertEqual(0, usage.call_count)
        self.assertEqual(0, usage.total_tokens)
        self.assertIn("Cache hits: 1", report)
        self.assertIn("Validation failures: 1", report)

    def test_report_includes_output_digest_and_pricing_basis(self):
        report = render_usage_report(UsageTracker(), 1, output_digest="abc123")
        self.assertIn("Output CSV SHA-256: abc123", report)
        self.assertIn("https://developers.openai.com/api/docs/models/gpt-5-mini", report)
        self.assertIn("Cost formula:", report)

    def test_report_includes_safe_validation_diagnostics(self):
        usage = UsageTracker()
        usage.record_validation_failure(
            "image", "image_1", "request_1", "image currency EUR conflicts with USD", 1,
        )
        report = render_usage_report(usage, 1)
        self.assertIn("Evidence validation diagnostics", report)
        self.assertIn("| image | image_1 | request_1 | 1 | image currency EUR conflicts with USD |", report)


if __name__ == "__main__":
    unittest.main()
