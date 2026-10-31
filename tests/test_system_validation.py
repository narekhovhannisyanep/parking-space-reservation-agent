"""Stage 4 harness contracts; load/live runs stay explicit, not part of discovery."""

import io
import json
import os
import sys
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from langchain_core.messages import HumanMessage

from context_augmentation import CONTEXT_HEADER
from parking_agent import AgentConfig
from recording import RecordingResult
from system_validation import (
    ScenarioRunner,
    ValidationFailure,
    ValidationServer,
    _serve,
    latency_summary,
    main,
    maximum_concurrency,
    run_validation,
    server_environment,
    synthetic_answer,
    synthetic_extraction,
)


class MetricTests(unittest.TestCase):
    def test_latency_reports_count_median_nearest_rank_p95_and_maximum(self):
        result = latency_summary([0.001, 0.003, 0.002])
        self.assertEqual(result, {"n": 3, "median_ms": 2.0, "p95_ms": 3.0, "max_ms": 3.0})
        self.assertEqual(latency_summary([]), {"n": 0})
        self.assertEqual(latency_summary([0]), {"n": 1, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0})

    def test_overlap_counts_execution_not_adjacent_or_zero_length_intervals(self):
        self.assertEqual(maximum_concurrency([]), 0)
        self.assertEqual(maximum_concurrency([(0, 1), (1, 2), (1, 1)]), 1)
        self.assertEqual(maximum_concurrency([(0, 3), (1, 2), (1.5, 4)]), 3)

    def test_invalid_timing_samples_fail_instead_of_producing_a_benchmark(self):
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                latency_summary([value])
        for interval in ((2, 1), (0, float("nan")), (0, float("inf"))):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                maximum_concurrency([interval])


class FixtureTests(unittest.TestCase):
    def test_extraction_is_verbatim_latest_turn_only_and_duration_only(self):
        messages = [
            HumanMessage("First name Old; surname Old; car number old-bike; reserve for 9 hours."),
            HumanMessage("First name Demo; surname Tester; car number stage4-bike-1; reserve for 1.5 hours."),
        ]
        value = synthetic_extraction(messages)
        self.assertEqual(value.intent, "reservation")
        self.assertEqual(
            {item.field: item.value for item in value.updates},
            {"first_name": "Demo", "surname": "Tester", "car_number": "stage4-bike-1", "duration_hours": "1.5 hours"},
        )
        self.assertEqual(synthetic_extraction([HumanMessage("What are the opening hours?")]).updates, [])
        correction = synthetic_extraction([HumanMessage("Change surname to Example.")])
        self.assertEqual(correction.updates[0].value, "Example")

    def test_concurrent_extractions_have_no_shared_response_queue_or_draft(self):
        def extract(index):
            return synthetic_extraction([HumanMessage(f"Car number stage4-bike-{index}.")]).updates[0].value

        with ThreadPoolExecutor(max_workers=5) as pool:
            self.assertEqual(list(pool.map(extract, range(20))), [f"stage4-bike-{index}" for index in range(20)])

    def test_duration_correction_does_not_need_a_new_availability_request(self):
        value = synthetic_extraction([HumanMessage("Change duration to 3 hours.")])
        self.assertEqual({item.field: item.value for item in value.updates}, {"duration_hours": "3 hours"})

    def test_unknown_part_or_invalid_profile_cannot_claim_a_validation_pass(self):
        with self.assertRaises(ValueError):
            run_validation("misspelled-part")
        for sessions, levels in ((0, (1,)), (31, (1,)), (3, ()), (3, (0,))):
            with self.subTest(sessions=sessions, levels=levels), self.assertRaises(ValueError):
                run_validation("load", sessions=sessions, concurrency=levels)

    def test_cli_failures_print_only_safe_codes_and_return_nonzero(self):
        output = io.StringIO()
        with (
            patch("system_validation.run_validation", side_effect=ValidationFailure("graph_run_error")),
            patch("sys.stdout", output),
        ):
            self.assertEqual(main(["--part", "integration"]), 1)
        self.assertEqual(json.loads(output.getvalue()), {"passed": False, "error_code": "graph_run_error"})

    def test_failed_part_keeps_completed_evidence_and_safe_receipt_diagnostics(self):
        receipt = RecordingResult(
            status="unconfirmed",
            request_id="private-identity",
            error_code="transport_error",
            retryable=True,
            attempts=2,
        )
        with (
            patch("system_validation.integration_validation", return_value={"cases_passed": 11}),
            patch(
                "system_validation.load_validation",
                side_effect=ValidationFailure("recording_not_confirmed", receipt=receipt),
            ),
        ):
            report = run_validation("offline", sessions=1, concurrency=(1,))
        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_part"], "load_1")
        self.assertEqual(report["integration"]["cases_passed"], 11)
        self.assertEqual(report["recording_failure"]["attempts"], 2)
        self.assertNotIn("private-identity", json.dumps(report))

    def test_cli_returns_nonzero_for_a_failed_report(self):
        with (
            patch("system_validation.run_validation", return_value={"passed": False}),
            patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(main(["--part", "integration"]), 1)

    def test_stub_answer_uses_current_evidence_not_history_and_rejects_private_input(self):
        context = {
            "dynamic_facts": {"prices": {"status": "snapshot", "value": 3}, "working_hours": {"status": "missing"}},
            "availability_options": [{"space_id": "G-A01"}],
        }
        answer = synthetic_answer([HumanMessage("What is available?" + CONTEXT_HEADER + json.dumps(context))])
        self.assertIn("3", answer.content)
        self.assertIn("unknown", answer.content)
        self.assertIn("G-A01", answer.content)
        self.assertNotIn("approved", answer.content)
        with self.assertRaises(ValidationFailure):
            synthetic_answer([HumanMessage("My car number is stage4-bike-99." + CONTEXT_HEADER + json.dumps(context))])

    def test_child_environment_is_disposable_and_offline_by_default(self):
        with (
            TemporaryDirectory(prefix="parking-stage4-") as directory,
            patch.dict(
                os.environ,
                {
                    "AZURE_OPENAI_API_KEY": "private-key",
                    "LANGSMITH_API_KEY": "private-trace-key",
                    "LANGSMITH_TRACING": "true",
                },
            ),
        ):
            environment = server_environment(Path(directory), live=False)
            self.assertNotIn("AZURE_OPENAI_API_KEY", environment)
            self.assertNotIn("LANGSMITH_API_KEY", environment)
            self.assertEqual(environment["LANGSMITH_TRACING"], "false")
            self.assertEqual(environment["LANGCHAIN_TRACING_V2"], "false")
            self.assertTrue(Path(environment["PARKING_SQLITE_PATH"]).is_relative_to(directory))
            self.assertTrue(Path(environment["PARKING_CHROMA_DIRECTORY"]).is_relative_to(directory))

    def test_live_environment_only_forwards_selected_provider_settings(self):
        with (
            TemporaryDirectory(prefix="parking-stage4-") as directory,
            patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "synthetic-key", "UNRELATED_SECRET": "do-not-forward"}),
        ):
            environment = server_environment(Path(directory), live=True)
            self.assertEqual(environment["AZURE_OPENAI_API_KEY"], "synthetic-key")
            self.assertNotIn("UNRELATED_SECRET", environment)
            self.assertNotIn("LANGSMITH_API_KEY", environment)


class ServerTests(unittest.TestCase):
    def test_bootstrap_requires_its_own_storage_settings_before_loading_the_server(self):
        application = SimpleNamespace(
            parking_agent=Mock(config=AgentConfig()),
            lifespan=Mock(),
            webapp=SimpleNamespace(router=SimpleNamespace(lifespan_context=None)),
        )
        with (
            TemporaryDirectory(prefix="parking-stage4-") as directory,
            patch.dict(os.environ, {}, clear=True),
            patch.dict(sys.modules, {"app": application}),
            patch("system_validation._instrument"),
            patch("langgraph_api.cli.run_server") as run,
        ):
            with self.assertRaises(ValidationFailure):
                _serve(Path(directory), 12345, 1, live=False)
            run.assert_not_called()

    def test_http_200_graph_errors_and_malformed_bodies_are_not_success(self):
        with TemporaryDirectory(prefix="parking-stage4-") as directory:
            server = ValidationServer(Path(directory))
            for contents in (b'{"__error__":{"message":"private-data"}}', b"[]", b"not json", b"x" * 2_000_001):
                response = MagicMock()
                response.__enter__.return_value.read.return_value = contents
                opener = Mock()
                opener.open.return_value = response
                with (
                    patch("urllib.request.build_opener", return_value=opener),
                    self.assertRaises(ValidationFailure) as failure,
                ):
                    server.request("GET", "/ok")
                self.assertNotIn("private-data", str(failure.exception))
                response.__exit__.assert_called_once()

    def test_http_error_response_handles_close_without_exposing_contents(self):
        with TemporaryDirectory(prefix="parking-stage4-") as directory:
            server = ValidationServer(Path(directory))
            body = io.BytesIO(b"private-data")
            error = urllib.error.HTTPError("http://127.0.0.1/ok", 503, "private-data", Message(), body)
            opener = Mock()
            opener.open.side_effect = error
            with (
                patch("urllib.request.build_opener", return_value=opener),
                self.assertRaises(ValidationFailure) as failure,
            ):
                server.request("GET", "/ok")
            self.assertEqual(str(failure.exception), "http_503")
            self.assertTrue(body.closed)

    def test_socket_server_runs_rag_multiturn_review_and_real_stdio_without_external_network(self):
        with TemporaryDirectory(prefix="parking-stage4-") as directory:
            server = ValidationServer(Path(directory))
            with server:
                runner = ScenarioRunner(server)
                approved = runner.conversation("approve-case", "approve", detailed=True)
                refused = runner.conversation("refuse-case", "refuse")
                self.assertEqual(approved["status"], "approved")
                self.assertEqual(refused["status"], "refused")
                self.assertEqual(runner.check_storage()["files"], 1)
                events = server.events()
                self.assertTrue(any(event["phase"] == "context_augmentation" for event in events))
                self.assertTrue(any(event["phase"] == "record_approved_reservation" for event in events))
                self.assertTrue(
                    all(set(event) == {"phase", "thread_id", "start", "end", "outcome"} for event in events)
                )
                self.assertNotIn("stage4-bike", json.dumps(events))
            assert server.process is not None
            self.assertIsNotNone(server.process.returncode)
            self.assertTrue(server.diagnostics.closed)

    def test_unmarked_workspace_and_invalid_settings_fail_before_launch(self):
        with TemporaryDirectory() as directory, self.assertRaises(ValueError):
            ValidationServer(Path(directory))
        with TemporaryDirectory(prefix="parking-stage4-") as directory:
            for jobs in (0, 11):
                with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                    ValidationServer(Path(directory), jobs=jobs)


if __name__ == "__main__":
    unittest.main()
