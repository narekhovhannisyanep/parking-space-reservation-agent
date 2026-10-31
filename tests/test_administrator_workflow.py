"""Offline Stage 2 submission and actual LangGraph interrupt/resume integration."""

import asyncio
import json
import sqlite3
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from openai import APIError
from pydantic import ValidationError
from test_recording_mcp import in_process_recorder
from test_reservation_storage import END, NOW, START, request, submitted_request
from test_static_storage import KeywordEmbeddings

from administrator_agent import AdministratorAgent, AdministratorDecision, ReviewWording
from parking_agent import AgentConfig, ParkingSpaceReservationAgent, ParkingState
from reservation_collection import FieldUpdate, ReservationExtraction
from storage import StorageConfig


class AdministratorAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.composer = Mock(spec=Runnable)
        self.composer.invoke.return_value = ReviewWording(message="Please review this reservation request.")
        self.composer.ainvoke = AsyncMock(return_value=self.composer.invoke.return_value)
        self.admin = AdministratorAgent(composer_factory=lambda: self.composer)

    def test_request_payload_uses_canonical_fields_not_model_authority(self):
        payload = self.admin.prepare(submitted_request(), "G-A01", "draft-1:1", {})
        self.assertEqual(payload["request"]["first_name"], "Demo")
        self.assertEqual(payload["request"]["car_number"], "my bicycle")
        self.assertEqual(payload["space_id"], "G-A01")
        self.assertEqual(payload["status"], "pending")
        self.assertEqual(payload["request_id"], "draft-1:1")
        self.assertIn("Resume", payload["instructions"])
        self.assertNotIn("copy", payload["instructions"].lower())
        self.assertNotIn("request_id", payload["instructions"])
        self.assertNotIn("approved", payload)
        prompt = self.composer.invoke.call_args.args[0]
        self.assertEqual(len(prompt), 2)
        self.assertIn("human administrator", prompt[0].content)
        self.assertNotIn("conversation", prompt[-1].content)
        self.assertIn("nostream", self.composer.invoke.call_args.kwargs["config"]["tags"])

    def test_unsafe_or_authoritative_wording_cannot_replace_request_or_approve(self):
        for output in (
            None,
            {"message": "Approved and recorded!"},
            {"message": "Please review Demo Tester my bicycle"},
            {"message": "Review", "approved": True},
        ):
            with self.subTest(output=output):
                self.composer.invoke.return_value = output
                payload = self.admin.prepare(submitted_request(), "G-A01", "draft-1:1", {})
                self.assertEqual(payload["message"], "Please approve or refuse the reservation request below.")
                self.assertEqual(payload["status"], "pending")

    def test_interface_instructions_are_code_owned_not_generated_wording(self):
        payload = self.admin.prepare(submitted_request(), "G-A01", "draft-1:1", {})
        prompt = self.composer.invoke.call_args.args[0][0].content
        self.assertIn("Do not name the review interface", prompt)
        self.assertNotIn("Studio", prompt)
        self.assertIn("Decision", payload["instructions"])
        self.assertIn("Resume", payload["instructions"])

    def test_decision_schema_requires_only_an_explicit_human_decision(self):
        schema = AdministratorDecision.model_json_schema()
        self.assertEqual(schema["required"], ["decision"])
        self.assertEqual(set(schema["properties"]), {"decision"})
        self.assertNotIn("default", schema["properties"]["decision"])
        for value in ("approve", "refuse"):
            self.assertEqual(AdministratorDecision.model_validate({"decision": value}).decision, value)
        for data in (
            {},
            {"decision": None},
            {"decision": True},
            {"decision": "yes"},
            {"decision": "approve", "request_id": "x"},
            {"decision": "approve", "approved": True},
        ):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                AdministratorDecision.model_validate(data)

    async def test_async_preparation_uses_native_async_composer(self):
        payload = await self.admin.aprepare(submitted_request(), "G-A01", "draft-1:1", {})
        self.assertEqual(payload["status"], "pending")
        self.composer.invoke.assert_not_called()
        self.composer.ainvoke.assert_awaited_once()

    def test_provider_failure_preserves_review_with_deterministic_wording(self):
        self.composer.invoke.side_effect = APIError("private-api-key", request=Mock(), body=None)
        payload = self.admin.prepare(submitted_request(), "G-A01", "draft-1:1", {})
        self.assertEqual(payload["message"], "Please approve or refuse the reservation request below.")
        self.assertNotIn("private-api-key", str(payload))


class AdministratorWorkflowFixture(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.storage = StorageConfig(chroma_directory=root / "chroma", sqlite_path=root / "parking.sqlite3")
        self.now = NOW
        self.extraction = ReservationExtraction()
        self.composer = Mock(spec=Runnable)
        self.composer.invoke.return_value = ReviewWording(message="Please review this reservation request.")
        self.composer.ainvoke = AsyncMock(return_value=self.composer.invoke.return_value)
        self.model = Mock()
        self.model.invoke.return_value = AIMessage("Parking information")
        self.model.ainvoke = AsyncMock(return_value=AIMessage("Parking information"))
        self.agent = ParkingSpaceReservationAgent(
            config=AgentConfig(storage=self.storage),
            embeddings=KeywordEmbeddings(),
            llm_client=self.model,
            reservation_extractor=RunnableLambda(lambda _: self.extraction),
            administrator_composer=self.composer,
            checkpointer=InMemorySaver(),
            clock=lambda: self.now,
            require_startup=True,
            recording_client=in_process_recorder(self.storage),
        )
        self.agent.prepare_startup()
        self.config: RunnableConfig = {"configurable": {"thread_id": "one"}}

    def record(self, identifier, thread="one"):
        row = self.agent.reservation_store.get(identifier, thread)
        assert row is not None
        return row

    def turn(self, text, config=None):
        return self.agent.graph.invoke({"messages": [HumanMessage(text)]}, config or self.config)

    def state(self, config=None):
        return self.agent.graph.get_state(config or self.config).values

    def fill(self, config=None, duration_hours="2"):
        fields = {
            "first_name": "Demo",
            "surname": "Tester",
            "car_number": "my bicycle",
            "duration_hours": duration_hours,
        }
        self.extraction = ReservationExtraction(
            intent="reservation",
            updates=[FieldUpdate.model_validate({"field": k, "value": v}) for k, v in fields.items()],
        )
        self.turn("Reserve: " + "; ".join(fields.values()), config)
        self.extraction = ReservationExtraction()
        self.turn(f"/confirm {self.state(config)['reservation']['revision']}", config)

    def options(self, config=None):
        self.turn("/options", config)
        return self.state(config)["offered_space_ids"]

    def submit(self, config=None):
        self.fill(config)
        offered = self.options(config)
        result = self.turn(f"/submit {offered[0]}", config)
        return result["__interrupt__"][0], offered[0]

    def decide(self, interrupt, decision="approve", config=None):
        return self.agent.graph.invoke(Command(resume={interrupt.id: {"decision": decision}}), config or self.config)


class AdministratorWorkflowTests(AdministratorWorkflowFixture):
    def test_confirmation_and_options_never_submit_or_hold(self):
        self.fill()
        self.options()
        self.assertFalse(self.state().get("submission"))
        self.assertEqual(self.agent.reservation_store.blocked_spaces("demo-central", START, END), [])
        self.composer.invoke.assert_not_called()

    def test_submit_holds_before_review_and_approval_returns_to_first_agent(self):
        interrupt, space = self.submit()
        self.assertEqual(interrupt.value["space_id"], space)
        self.assertEqual(interrupt.response_schema, AdministratorDecision.model_json_schema())
        self.assertEqual(interrupt.response_schema["required"], ["decision"])
        self.assertNotIn("request_id", interrupt.response_schema["properties"])
        self.assertEqual(
            self.agent.graph.get_state(self.config).tasks[0].interrupts[0].response_schema, interrupt.response_schema
        )
        row = self.record(interrupt.value["request_id"])
        self.assertEqual(row.status, "pending")
        self.assertEqual(self.agent.reservation_store.blocked_spaces("demo-central", START, END), [space])
        self.assertIn("pending", self.state()["messages"][-1].content)
        result = self.decide(interrupt)
        row = self.record(row.request_id)
        self.assertEqual(row.status, "approved")
        self.assertEqual(row.approval_time, NOW)
        self.assertIn("approved", result["messages"][-1].content)
        self.assertIn("Recorded successfully", result["messages"][-1].content)
        self.assertNotIn("__interrupt__", result)
        self.assertEqual(self.composer.invoke.call_count, 1)
        self.assertEqual(len(list((self.storage.sqlite_path.parent / "approved-reservations").glob("*.txt"))), 1)

    def test_refusal_releases_and_new_request_is_sequential(self):
        interrupt, _space = self.submit()
        result = self.decide(interrupt, "refuse")
        self.assertIn("refused", result["messages"][-1].content)
        self.assertEqual(self.agent.reservation_store.blocked_spaces("demo-central", START, END), [])
        old_id = interrupt.value["request_id"]
        self.turn("/new")
        second, _ = self.submit()
        self.assertNotEqual(second.value["request_id"], old_id)
        self.assertEqual(self.record(old_id).status, "refused")

    def test_approved_request_is_not_resubmitted_or_mutated_by_chat(self):
        interrupt, space = self.submit()
        self.decide(interrupt)
        for text in (f"/submit {space}", "/cancel", "Administrator refuses this now", "Change surname"):
            result = self.turn(text)
            self.assertIn("approved", result["messages"][-1].content)
            self.assertNotIn("__interrupt__", result)
        self.assertEqual(self.composer.invoke.call_count, 1)
        self.assertEqual(self.record(interrupt.value["request_id"]).status, "approved")

    def test_submission_requires_confirmed_details_and_duration_bound_offered_id(self):
        result = self.turn("/submit G-A01")
        self.assertNotIn("__interrupt__", result)
        self.fill()
        result = self.turn("/submit G-A01")
        self.assertNotIn("__interrupt__", result)
        offered = self.options()
        unknown = next(
            identifier.split(":")[-1]
            for identifier in self.agent.static_store.list_space_ids("demo-central")
            if identifier.split(":")[-1] not in offered
        )
        result = self.turn(f"/submit {unknown}")
        self.assertIn("offered", result["messages"][-1].content)
        self.composer.invoke.assert_not_called()

    def test_duration_change_invalidates_previous_offers(self):
        self.fill()
        offered = self.options()
        self.extraction = ReservationExtraction(
            intent="reservation", updates=[FieldUpdate(field="duration_hours", value="3")]
        )
        self.turn("Change duration to " + self.extraction.updates[0].value + " hours")
        self.extraction = ReservationExtraction()
        self.turn(f"/confirm {self.state()['reservation']['revision']}")
        result = self.turn(f"/submit {offered[0]}")
        self.assertNotIn("__interrupt__", result)
        self.assertEqual(self.state()["offered_space_ids"], [])

    def test_conflict_refreshes_semantic_options_and_does_not_contact_admin(self):
        self.fill()
        offered = self.options()
        ids = self.agent.static_store.list_space_ids("demo-central")
        self.agent.reservation_store.hold(request("competitor"), offered[0], "other", ids)
        result = self.turn(f"/submit {offered[0]}")
        self.assertNotIn("__interrupt__", result)
        self.assertIn("overlapping", result["messages"][-1].content)
        self.assertNotIn(offered[0], self.state()["offered_space_ids"])
        self.assertEqual(len(self.state()["offered_space_ids"]), 3)
        self.composer.invoke.assert_not_called()

    def test_other_thread_period_suggestions_exclude_pending_and_allow_adjacent(self):
        interrupt, space = self.submit()
        other = {"configurable": {"thread_id": "two"}}
        self.fill(other)
        self.assertNotIn(space, self.options(other))
        self.turn("/new", other)
        self.now = END
        self.fill(other, duration_hours="1")
        with patch.object(self.agent.static_store, "search", wraps=self.agent.static_store.search) as search:
            self.options(other)
        self.assertIn(f"demo-central:space:{space}", search.call_args.kwargs["eligible_ids"])
        self.assertEqual(self.record(interrupt.value["request_id"]).status, "pending")

    def test_invalid_decision_only_responses_never_approve_and_can_retry(self):
        interrupt, _ = self.submit()
        for data in (
            {},
            {"decision": None},
            {"decision": True},
            {"decision": "yes"},
            {"request_id": "another-request", "decision": "approve"},
        ):
            with self.subTest(data=data):
                result = self.agent.graph.invoke(Command(resume={interrupt.id: data}), self.config)
                self.assertIn("__interrupt__", result)
                interrupt = result["__interrupt__"][0]
                self.assertIn("validation_error", interrupt.value)
                self.assertNotIn("request_id", interrupt.value["validation_error"])
                self.assertEqual(interrupt.response_schema["required"], ["decision"])
            self.assertEqual(self.record(interrupt.value["request_id"]).status, "pending")
        result = self.decide(interrupt)
        self.assertIn("approved", result["messages"][-1].content)
        self.assertEqual(self.composer.invoke.call_count, 1)

    def test_review_payload_identity_is_checked_automatically_sync_and_async(self):
        interrupt, _ = self.submit()
        original = self.record(interrupt.value["request_id"])
        for payload_id in (None, "another-request"):
            for asynchronous in (False, True):
                with self.subTest(payload_id=payload_id, asynchronous=asynchronous):
                    state = ParkingState(**self.state())
                    state.administrator_payload = {**state.administrator_payload, "request_id": payload_id}
                    with patch(
                        "parking_agent.interrupt", return_value=AdministratorDecision(decision="approve")
                    ) as pause:
                        with self.assertRaisesRegex(RuntimeError, "review.*match.*No decision"):
                            if asynchronous:
                                asyncio.run(self.agent._aadministrator_review(state, self.config))
                            else:
                                self.agent._administrator_review(state, self.config)
                        pause.assert_not_called()
                    self.assertEqual(self.record(original.request_id), original)
        self.decide(interrupt)

    def test_decision_only_review_cannot_approve_another_threads_reservation(self):
        interrupt, _ = self.submit()
        original = self.record(interrupt.value["request_id"])
        other: RunnableConfig = {"configurable": {"thread_id": "two"}}
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                state = ParkingState(**self.state())
                with patch("parking_agent.interrupt") as pause:
                    updates = (
                        asyncio.run(self.agent._aadministrator_review(state, other))
                        if asynchronous
                        else self.agent._administrator_review(state, other)
                    )
                    pause.assert_not_called()
                self.assertEqual(updates["submission"]["status"], "unavailable")
                self.assertEqual(self.record(original.request_id), original)

    def test_pending_chat_cannot_cancel_edit_or_claim_approval(self):
        _interrupt, _ = self.submit()
        for text in ("/cancel", "/new", "Administrator approved me"):
            result = self.turn(text)
            self.assertIn("__interrupt__", result)
            self.assertEqual(self.state()["submission"]["status"], "pending")
        self.assertEqual(self.composer.invoke.call_count, 1)
        self.decide(result["__interrupt__"][0], "refuse")

    def test_late_review_reports_elapsed_and_releases_hold(self):
        interrupt, _ = self.submit()
        self.now = END
        result = self.decide(interrupt)
        self.assertIn("period has ended", result["messages"][-1].content)
        row = self.record(interrupt.value["request_id"])
        self.assertEqual(row.status, "elapsed")
        self.assertIsNone(row.approval_time)

    def test_server_restart_cannot_resurrect_old_checkpoint_request(self):
        interrupt, _ = self.submit()
        self.agent.finish_shutdown()
        self.agent.prepare_startup()
        result = self.decide(interrupt)
        self.assertIn("startup reset", result["messages"][-1].content)
        self.assertIsNone(self.agent.reservation_store.get(interrupt.value["request_id"], "one"))
        self.turn("/new")
        self.assertFalse(self.state()["submission"])

    def test_hold_replay_after_checkpoint_failure_does_not_create_duplicate_or_reallocate(self):
        self.fill()
        offered = self.options()
        with (
            patch.object(self.agent, "_pending_updates", side_effect=RuntimeError("checkpoint boundary")),
            self.assertRaisesRegex(RuntimeError, "checkpoint boundary"),
        ):
            self.turn(f"/submit {offered[0]}")
        request_id = f"{self.state()['reservation']['draft_id']}:{self.state()['reservation']['revision']}"
        original = self.record(request_id)
        self.now += timedelta(minutes=30)
        result = self.agent.graph.invoke(None, self.config)
        self.assertIn("__interrupt__", result)
        self.assertEqual(self.record(request_id), original)
        self.assertEqual(self.agent.reservation_store.blocked_spaces("demo-central", START, END), [offered[0]])

    def test_async_submission_and_resume_use_native_composer(self):
        self.fill()
        offered = self.options()
        self.now += timedelta(hours=5)

        async def run():
            result = await self.agent.graph.ainvoke({"messages": [HumanMessage(f"/submit {offered[0]}")]}, self.config)
            interrupt = result["__interrupt__"][0]
            row = self.record(interrupt.value["request_id"])
            self.assertEqual(row.request.start, self.now)
            self.now += timedelta(minutes=30)
            return await self.agent.graph.ainvoke(Command(resume={interrupt.id: {"decision": "approve"}}), self.config)

        result = asyncio.run(run())
        self.assertIn("approved", result["messages"][-1].content)
        self.composer.invoke.assert_not_called()
        self.composer.ainvoke.assert_awaited_once()
        row = self.record(self.state()["submission"]["request_id"])
        self.assertEqual(row.request.end - row.request.start, timedelta(hours=2))
        self.assertEqual(row.approval_time, self.now)

    def test_async_missing_decision_reinterrupts_without_approval_or_recomposition(self):
        self.fill()
        offered = self.options()

        async def run():
            result = await self.agent.graph.ainvoke({"messages": [HumanMessage(f"/submit {offered[0]}")]}, self.config)
            pending = result["__interrupt__"][0]
            original = self.record(pending.value["request_id"])
            invalid = await self.agent.graph.ainvoke(Command(resume={pending.id: {}}), self.config)
            retry = invalid["__interrupt__"][0]
            self.assertIn("validation_error", retry.value)
            self.assertEqual(self.record(original.request_id), original)
            self.assertEqual(retry.response_schema["required"], ["decision"])
            return await self.agent.graph.ainvoke(Command(resume={retry.id: {"decision": "refuse"}}), self.config)

        result = asyncio.run(run())
        self.assertIn("refused", result["messages"][-1].content)
        self.assertEqual(self.agent.reservation_store.blocked_spaces("demo-central", START, END), [])
        self.composer.invoke.assert_not_called()
        self.assertEqual(self.composer.ainvoke.await_count, 1)

    def test_raw_administrator_model_tokens_are_not_published_sync_or_async(self):
        payload = json.dumps({"message": "Please review Demo Tester my bicycle password=synthetic-secret"})
        self.agent._administrator_composer = FakeListChatModel(responses=[payload]) | RunnableLambda(
            lambda message: ReviewWording.model_validate_json(message.content)
        )

        def check(chunks):
            self.assertTrue(chunks)
            for message, _ in chunks:
                assert isinstance(message, AIMessage)
                self.assertNotIn("synthetic-secret", message.content)
                self.assertNotIn('"message"', message.content)
                self.assertNotIn("my bicycle", message.content)

        self.fill()
        offered = self.options()
        check(
            list(
                self.agent.graph.stream(
                    {"messages": [HumanMessage(f"/submit {offered[0]}")]}, self.config, stream_mode="messages"
                )
            )
        )
        pending = self.agent.graph.get_state(self.config).tasks[0].interrupts[0]
        self.assertEqual(pending.value["message"], "Please approve or refuse the reservation request below.")
        self.decide(pending, "refuse")
        self.turn("/new")
        self.fill()
        offered = self.options()

        async def run():
            return [
                chunk
                async for chunk in self.agent.graph.astream(
                    {"messages": [HumanMessage(f"/submit {offered[0]}")]}, self.config, stream_mode="messages"
                )
            ]

        check(asyncio.run(run()))

    def test_no_checkpointer_rejects_submission_before_creating_a_hold(self):
        self.fill()
        offered = self.options()
        state = self.state()
        graph = self.agent._build_graph(None)
        with patch.object(self.agent, "graph", graph), self.assertRaisesRegex(ValueError, "checkpointer"):
            self.agent._operational_input(
                ParkingState(**state),
                f"/submit {offered[0]}",
                self.config,
            )
        self.assertEqual(self.agent.reservation_store.blocked_spaces("demo-central", START, END), [])

    def test_storage_errors_fail_closed_without_private_error_text(self):
        self.fill()
        offered = self.options()
        with (
            patch.object(self.agent.reservation_store, "hold", side_effect=sqlite3.OperationalError("private/path")),
            self.assertRaisesRegex(RuntimeError, "Unable to read or submit") as error,
        ):
            self.turn(f"/submit {offered[0]}")
        self.assertNotIn("private/path", str(error.exception))
        self.composer.invoke.assert_not_called()
        with (
            patch.object(
                self.agent.reservation_store, "blocked_spaces", side_effect=sqlite3.OperationalError("private/path")
            ),
            self.assertRaisesRegex(RuntimeError, "Unable to retrieve") as error,
        ):
            self.turn("/options")
        self.assertNotIn("private/path", str(error.exception))

    def test_decision_write_replay_preserves_approval_after_response_failure(self):
        interrupt, _ = self.submit()
        with (
            patch.object(self.agent, "_outcome_reply", side_effect=RuntimeError("response boundary")),
            self.assertRaisesRegex(RuntimeError, "response boundary"),
        ):
            self.decide(interrupt)
        row = self.record(interrupt.value["request_id"])
        self.assertEqual(row.status, "approved")
        self.now += timedelta(minutes=1)
        result = self.agent.graph.invoke(None, self.config)
        self.assertIn("approved", result["messages"][-1].content)
        self.assertEqual(self.record(row.request_id), row)

    def test_unknown_period_excludes_only_currently_active_holds(self):
        ids = self.agent.static_store.list_space_ids("demo-central")
        row = self.agent.reservation_store.hold(request("active"), "G-A01", "other", ids)
        self.now += timedelta(seconds=1)
        with patch.object(self.agent.static_store, "search", wraps=self.agent.static_store.search) as search:
            self.options()
        self.assertNotIn("demo-central:space:G-A01", search.call_args.kwargs["eligible_ids"])
        self.assertEqual(self.record(row.request_id, "other").status, "pending")


if __name__ == "__main__":
    unittest.main()
