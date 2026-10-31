"""Stage 3 LangGraph integration with real in-process MCP and offline agent models."""

import asyncio
import sqlite3
import unittest
from datetime import timedelta
from unittest.mock import AsyncMock, patch

from langgraph.types import Command
from test_administrator_workflow import AdministratorWorkflowFixture
from test_reservation_storage import END, NOW, START

from recording import RecordingResult, recording_filename


class RecordingWorkflowTests(AdministratorWorkflowFixture):
    def files(self):
        return list(self.agent.config.recording.directory.glob("*.txt"))

    def test_human_approval_then_recording_and_final_reply_have_distinct_outcomes(self):
        self.fill()
        self.options()
        self.assertEqual(self.files(), [])
        pending, space = self.submit()
        self.assertEqual(self.files(), [])
        self.assertEqual(self.state()["submission"]["status"], "pending")
        result = self.decide(pending)
        self.assertIn("Recorded successfully", result["messages"][-1].content)
        row = self.record(pending.value["request_id"])
        self.assertEqual(row.approval_time, NOW)
        self.assertEqual(self.state()["recording"]["status"], "recorded")
        self.assertEqual(self.files()[0].name, recording_filename(row.request_id))
        self.assertEqual(self.agent.reservation_store.blocked_spaces("demo-central", START, END), [space])
        self.assertNotIn("Demo Tester", result["messages"][-1].content)
        self.assertNotIn("my bicycle", result["messages"][-1].content)

    def test_refusal_and_elapsed_review_never_call_recording(self):
        for decision, elapsed in (("refuse", False), ("approve", True)):
            with self.subTest(decision=decision, elapsed=elapsed):
                self.turn("/new")
                pending, _ = self.submit()
                if elapsed:
                    self.now += timedelta(hours=2)
                with patch.object(self.agent.recording_client, "record") as record:
                    result = self.decide(pending, decision)
                    record.assert_not_called()
                self.assertNotIn("Recorded successfully", result["messages"][-1].content)
                self.assertEqual(self.files(), [])

    def test_transient_failure_exhaustion_keeps_approval_and_manual_retry_is_safe(self):
        pending, space = self.submit()
        with patch.object(
            self.agent.recording_client, "_call", AsyncMock(side_effect=TimeoutError("private-secret"))
        ) as call:
            result = self.decide(pending)
            self.assertEqual(call.await_count, 2)
        row = self.record(pending.value["request_id"])
        self.assertEqual(row.status, "approved")
        self.assertEqual(row.approval_time, NOW)
        self.assertEqual(self.state()["recording"]["status"], "unconfirmed")
        self.assertIn("could not be confirmed", result["messages"][-1].content)
        self.assertIn("/retry-recording", result["messages"][-1].content)
        self.assertNotIn("private-secret", str(self.state()["recording"]))
        self.assertEqual(self.agent.reservation_store.blocked_spaces("demo-central", START, END), [space])
        with patch.object(self.agent.recording_client, "record") as record:
            self.turn("Please show the status")
            record.assert_not_called()
        self.now += timedelta(minutes=1)
        result = self.turn("/retry-recording")
        self.assertIn("Recorded successfully", result["messages"][-1].content)
        self.assertEqual(len(self.files()), 1)
        self.assertEqual(self.record(row.request_id), row)
        self.assertEqual(self.composer.invoke.call_count, 1)
        with patch.object(self.agent.recording_client, "record") as record:
            self.turn("/retry-recording")
            record.assert_not_called()

    def test_permanent_write_failure_is_explicit_and_not_automatically_retried(self):
        pending, _ = self.submit()
        failure = RecordingResult(status="failed", request_id=pending.value["request_id"], error_code="storage_error")
        with patch.object(self.agent.recording_client, "_call", AsyncMock(return_value=failure)) as call:
            result = self.decide(pending)
        self.assertEqual(call.await_count, 1)
        self.assertIn("Recording failed", result["messages"][-1].content)
        self.assertNotIn("Recorded successfully", result["messages"][-1].content)
        self.assertEqual(self.record(pending.value["request_id"]).status, "approved")
        self.assertEqual(self.files(), [])

    def test_node_replay_after_file_write_does_not_duplicate_or_repeat_approval(self):
        pending, _ = self.submit()
        with (
            patch.object(self.agent, "_recording_updates", side_effect=RuntimeError("checkpoint boundary")),
            self.assertRaisesRegex(RuntimeError, "checkpoint boundary"),
        ):
            self.decide(pending)
        row = self.record(pending.value["request_id"])
        self.assertEqual(row.status, "approved")
        self.assertEqual(len(self.files()), 1)
        self.now = END + timedelta(days=1)
        result = self.agent.graph.invoke(None, self.config)
        self.assertIn("Recorded successfully", result["messages"][-1].content)
        self.assertEqual(len(self.files()), 1)
        self.assertEqual(self.record(row.request_id), row)
        self.assertEqual(self.composer.invoke.call_count, 1)

    def test_retry_requires_current_human_approval_and_new_does_not_reuse_receipt(self):
        with patch.object(self.agent.recording_client, "record") as record:
            self.turn("/retry-recording")
            record.assert_not_called()
        pending, _ = self.submit()
        with patch.object(self.agent.recording_client, "record") as record:
            result = self.turn("/retry-recording")
            record.assert_not_called()
        self.assertEqual(self.record(pending.value["request_id"]).status, "pending")
        self.decide(result["__interrupt__"][0])
        old = self.state()["recording"]
        self.turn("/new")
        self.assertEqual(self.state()["recording"], {})
        self.assertFalse(self.state()["recording_requested"])
        second, _ = self.submit()
        self.decide(second)
        self.assertNotEqual(self.state()["recording"]["request_id"], old["request_id"])
        self.assertEqual(len(self.files()), 2)

    def test_old_checkpoints_after_restart_cannot_resurrect_recording(self):
        pending, _ = self.submit()
        self.decide(pending)
        self.assertEqual(len(self.files()), 1)
        self.agent.finish_shutdown()
        self.agent.prepare_startup()
        self.assertEqual(self.files(), [])
        with patch.object(self.agent.recording_client, "record") as record:
            result = self.turn("/retry-recording")
            record.assert_not_called()
        self.assertIn("startup reset", result["messages"][-1].content)
        self.assertEqual(self.files(), [])

    def test_async_approval_and_retry_use_native_async_recording(self):
        pending, _ = self.submit()

        async def run():
            with patch.object(self.agent.recording_client, "record", side_effect=AssertionError("Use async recording")):
                return await self.agent.graph.ainvoke(
                    Command(resume={pending.id: {"decision": "approve"}}), self.config
                )

        result = asyncio.run(run())
        self.assertIn("Recorded successfully", result["messages"][-1].content)
        self.assertEqual(len(self.files()), 1)

    def test_post_approval_ledger_read_failure_does_not_claim_no_decision_was_applied(self):
        pending, _ = self.submit()
        original = self.agent.reservation_store.get

        def unavailable_after_approval(*args):
            row = original(*args)
            if row is not None and row.status == "approved":
                raise sqlite3.OperationalError("private SQL failure")
            return row

        with (
            patch.object(self.agent.reservation_store, "get", side_effect=unavailable_after_approval),
            patch.object(self.agent.recording_client, "record") as record,
        ):
            result = self.decide(pending)
            record.assert_not_called()
        self.assertEqual(self.record(pending.value["request_id"]).status, "approved")
        self.assertIn("Recording failed", result["messages"][-1].content)
        self.assertNotIn("no decision has been applied", result["messages"][-1].content.lower())
        self.assertNotIn("private SQL", result["messages"][-1].content)
        self.assertEqual(self.state()["recording"]["status"], "failed")
        self.assertEqual(self.files(), [])
        self.assertIn("Recorded successfully", self.turn("/retry-recording")["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
