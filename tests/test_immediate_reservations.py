"""Duration-only collection and server-timed holds, with no provider calls."""

import unittest
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

import reservation_collection as collection
from administrator_agent import ReviewWording
from guardrails import PrivacyGuard
from parking_agent import AgentConfig, ParkingSpaceReservationAgent
from storage import DynamicParkingStore, ParkingSpace, StaticParkingStore, StorageConfig
from storage.reservations import ReservationConflict, ReservationStore

NOW = datetime(2030, 1, 1, 23, 30, tzinfo=UTC)
DETAILS = {"first_name": "Demo", "surname": "Tester", "car_number": "my bicycle", "duration_hours": "2"}


def draft(duration: str | None = "2"):
    return collection.ReservationSession(
        draft_id="duration-draft",
        revision=1,
        status="collecting",
        draft=collection.ReservationDraft(**{**DETAILS, "duration_hours": duration}),
    )


def validated(duration="2"):
    request, issues = collection.validate_draft(draft(duration), "demo-central", NOW)
    assert request is not None and not issues
    return request


class DurationContractTests(unittest.TestCase):
    def test_collects_only_names_vehicle_and_hours_without_capturing_start(self):
        request, issues = collection.validate_draft(draft(), "demo-central", NOW)
        self.assertEqual(issues, [])
        assert request is not None
        self.assertEqual(request.duration_hours, 2)
        self.assertNotIn("start", request.model_dump())
        self.assertNotIn("timezone", request.model_dump())
        later, issues = collection.validate_draft(draft(), "demo-central", NOW + timedelta(days=1))
        self.assertEqual(later, request)
        summary = collection.confirmation_summary(request)
        self.assertIn("2 hours", summary)
        self.assertIn("submit", summary)
        self.assertNotIn("Timezone:", summary)

    def test_duration_parsing_and_invalid_input_are_explicit(self):
        for value, expected in (("2", 2), ("1.5 hours", 1.5), (".5 h", 0.5), ("two hours", 2)):
            with self.subTest(value=value):
                request, issues = collection.validate_draft(draft(value), "demo-central", NOW)
                self.assertEqual(issues, [])
                assert request is not None
                self.assertEqual(request.duration_hours, expected)
        for value in (
            None,
            "0",
            "-2",
            "NaN",
            "inf",
            "2 or 3",
            "tomorrow",
            "30 minutes",
            "1e100",
            "9" * 200,
            "0.000000000001",
        ):
            with self.subTest(value=value):
                request, issues = collection.validate_draft(draft(value), "demo-central", NOW)
                self.assertIsNone(request)
                self.assertTrue(issues)
                self.assertNotIn("timezone", " ".join(issues).lower())

    def test_model_cannot_supply_a_start_end_or_timezone(self):
        for field in ("start", "end", "timezone"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                collection.FieldUpdate.model_validate({"field": field, "value": "2030-01-01"})

    def test_legacy_checkpoint_discards_schedule_and_confirmation_not_personal_details(self):
        old = {
            "draft_id": "legacy",
            "revision": 3,
            "status": "ready",
            "confirmed_revision": 3,
            "summary_message_id": "old-summary",
            "request": {},
            "draft": {
                "first_name": "Demo",
                "surname": "Tester",
                "car_number": "my bicycle",
                "start": "2030-01-01 10:00",
                "end": "2030-01-01 12:00",
                "timezone": "UTC",
            },
        }
        session = collection.ReservationSession.model_validate(old)
        self.assertEqual(session.status, "collecting")
        self.assertEqual(session.draft.first_name, "Demo")
        self.assertIsNone(session.draft.duration_hours)
        self.assertIsNone(session.request)
        self.assertIsNone(session.confirmed_revision)
        self.assertGreater(session.revision, 3)


class ImmediateReservationTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        config = StorageConfig(chroma_directory=root / "chroma", sqlite_path=root / "parking.sqlite3")
        self.now = NOW
        guard = PrivacyGuard(lambda _: [])
        self.dynamic = DynamicParkingStore(config, privacy_guard=guard)
        self.store = ReservationStore(self.dynamic, clock=lambda: self.now)
        self.ids = ["demo-central:space:G-A01", "demo-central:space:G-A02"]
        self.static = Mock(spec=StaticParkingStore)
        self.static.list_space_ids.return_value = self.ids
        documents = [
            ParkingSpace(
                space_id=x.split(":")[-1],
                location_id="demo-central",
                label="Demo",
                description="Beside the entrance.",
                source="fixture",
            ).to_document()
            for x in self.ids
        ]
        self.static.search.side_effect = lambda *args, **kwargs: [
            doc for doc in documents if doc.id in kwargs.get("eligible_ids", self.ids)
        ][: kwargs["k"]]
        self.extraction = collection.ReservationExtraction()
        self.agent = ParkingSpaceReservationAgent(
            config=AgentConfig(storage=config),
            static_store=self.static,
            dynamic_store=self.dynamic,
            reservation_store=self.store,
            clock=lambda: self.now,
            privacy_guard=guard,
            reservation_extractor=RunnableLambda(lambda _: self.extraction),
            checkpointer=InMemorySaver(),
            administrator_composer=RunnableLambda(lambda _: ReviewWording(message="Please review this request.")),
        )
        self.config: RunnableConfig = {"configurable": {"thread_id": "one"}}

    def turn(self, text):
        return self.agent.graph.invoke({"messages": [HumanMessage(text)]}, self.config)

    def state(self):
        return self.agent.graph.get_state(self.config).values

    def fill(self):
        self.extraction = collection.ReservationExtraction(
            intent="reservation",
            updates=[collection.FieldUpdate.model_validate({"field": k, "value": v}) for k, v in DETAILS.items()],
        )
        self.turn("Reserve: " + "; ".join(DETAILS.values()))
        self.extraction = collection.ReservationExtraction()
        self.turn(f"/confirm {self.state()['reservation']['revision']}")
        self.turn("/options")
        return self.state()["offered_space_ids"][0]

    def test_missing_details_submit_starts_collection_and_names_the_missing_fields(self):
        reply = self.turn("/submit G-A01")["messages"][-1].content
        for phrase in ("first name", "surname", "vehicle identifier", "how many hours"):
            self.assertIn(phrase, reply.lower())
        self.assertNotIn("User confirmation is not administrator approval", reply)
        self.assertEqual(self.state()["reservation"]["status"], "collecting")
        self.assertFalse(self.state().get("submission"))

    def test_partial_details_submit_asks_only_for_missing_fields(self):
        self.extraction = collection.ReservationExtraction(
            intent="reservation",
            updates=[
                collection.FieldUpdate(field="first_name", value="Demo"),
                collection.FieldUpdate(field="surname", value="Tester"),
            ],
        )
        self.turn("Reserve for Demo Tester")
        reply = self.turn("/submit G-A01")["messages"][-1].content
        self.assertIn("vehicle identifier", reply)
        self.assertIn("how many hours", reply)
        self.assertNotIn("Provide first name", reply)
        self.assertNotIn("Provide surname", reply)
        self.assertFalse(self.state().get("submission"))

    def test_submit_complete_but_unconfirmed_details_displays_summary_without_holding(self):
        self.fill()
        self.turn("no")
        reply = self.turn("/submit G-A01")["messages"][-1].content
        session = collection.ReservationSession.model_validate(self.state()["reservation"])
        assert session.request is not None
        self.assertEqual(reply, collection.confirmation_summary(session.request))
        self.assertEqual(self.store.blocked_spaces("demo-central"), [])
        self.assertFalse(self.state().get("submission"))
        self.turn("yes")
        self.assertEqual(self.state()["reservation"]["status"], "ready")

    def test_delay_between_confirmation_options_and_submit_does_not_shorten_duration(self):
        space = self.fill()
        confirmed = self.state()["reservation"]
        self.now += timedelta(hours=5)
        result = self.turn(f"/submit {space}")
        payload = result["__interrupt__"][0].value
        row = self.store.get(payload["request_id"], "one")
        assert row is not None
        self.assertEqual(row.request.start, self.now)
        self.assertEqual(row.request.end, self.now + timedelta(hours=2))
        self.assertEqual(self.state()["reservation"], confirmed)
        self.assertIn(row.request.end.isoformat(), self.state()["messages"][-1].content)
        self.now += timedelta(minutes=30)
        self.agent.graph.invoke(Command(resume={"decision": "approve"}), self.config)
        approved = self.store.get(row.request_id, "one")
        assert approved is not None
        self.assertEqual(approved.status, "approved")
        self.assertEqual(approved.request, row.request)
        self.assertEqual(approved.approval_time, self.now)

    def test_hold_replay_never_moves_the_period_and_different_duration_is_rejected(self):
        request = validated()
        row = self.store.hold(request, "G-A01", "one", self.ids)
        self.now += timedelta(hours=3)
        self.assertEqual(self.store.hold(request, "G-A01", "one", self.ids), row)
        changed = validated("3")
        with self.assertRaises(ReservationConflict):
            self.store.hold(changed, "G-A01", "one", self.ids)

    def test_approval_at_end_is_elapsed_and_releases_space(self):
        request = validated()
        row = self.store.hold(request, "G-A01", "one", self.ids)
        self.now = row.request.end
        self.assertEqual(self.store.decide(row.request_id, "one", "approve").status, "elapsed")
        self.assertEqual(self.store.blocked_spaces("demo-central"), [])
        second = self.store.hold(request.model_copy(update={"draft_id": "second"}), "G-A01", "one", self.ids)
        self.assertEqual(second.request.start, row.request.end)

    def test_server_offset_normalizes_to_utc_and_naive_or_overflow_clock_fails_without_hold(self):
        request = validated()
        self.now = NOW.astimezone(timezone(timedelta(hours=4)))
        row = self.store.hold(request, "G-A01", "one", self.ids)
        self.assertEqual(row.request.start.tzinfo, UTC)
        self.assertEqual(row.request.start, NOW)
        for now in (NOW.replace(tzinfo=None), datetime.max.replace(tzinfo=UTC)):
            self.now = now
            with self.subTest(now=now), self.assertRaises(ValueError):
                self.store.hold(request.model_copy(update={"draft_id": "invalid-clock"}), "G-A02", "one", self.ids)
            self.assertIsNone(self.store.get("invalid-clock:1", "one"))

    def test_approval_just_before_end_retains_the_original_period(self):
        row = self.store.hold(validated(), "G-A01", "one", self.ids)
        self.now = row.request.end - timedelta(microseconds=1)
        approved = self.store.decide(row.request_id, "one", "approve")
        self.assertEqual(approved.status, "approved")
        self.assertEqual(approved.request, row.request)
        self.assertEqual(self.store.blocked_spaces("demo-central"), ["G-A01"])
