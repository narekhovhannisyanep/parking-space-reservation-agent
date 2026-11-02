"""Opt-in extraction/administrator smoke checks using synthetic details and mocked stores."""

import asyncio
import os
import unittest
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TypeVar
from unittest.mock import Mock

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from openai import APIError

from administrator_agent import AdministratorAgent, ReviewWording
from guardrails import guarded_model_config
from parking_agent import AgentConfig, ParkingSpaceReservationAgent
from pricing import resolve_budget
from reservation_collection import (
    EXTRACTION_FAILURE,
    READY_REPLY,
    ReservationSession,
    SubmittedReservationRequest,
    extraction_messages,
    parse_extraction,
)
from storage import DynamicParkingStore, StaticParkingStore
from storage.reservations import ReservationStore

Result = TypeVar("Result")


@unittest.skipUnless(os.getenv("PARKING_LIVE_TEST") == "1", "Set PARKING_LIVE_TEST=1 to call the real model.")
class LiveReservationModelTests(unittest.TestCase):
    def setUp(self):
        if (
            os.getenv("LANGSMITH_TRACING", "false").lower() != "false"
            or os.getenv("LANGCHAIN_TRACING_V2", "false").lower() != "false"
        ):
            self.skipTest("Disable model tracing for these smoke checks.")
        static = Mock(spec=StaticParkingStore)
        static.search.return_value = []
        static.list_space_ids.return_value = []
        static.get_spaces.return_value = []
        static.hourly_price_summary.return_value = {"status": "missing", "currency": "USD"}
        dynamic = Mock(spec=DynamicParkingStore)
        dynamic.list_facts.return_value = []
        ledger = Mock(spec=ReservationStore)
        ledger.blocked_spaces.return_value = []
        self.agent = ParkingSpaceReservationAgent(
            config=replace(AgentConfig.from_environment(), timeout=15, max_retries=0, reasoning_effort="low"),
            static_store=static,
            dynamic_store=dynamic,
            reservation_store=ledger,
            checkpointer=InMemorySaver(),
            clock=lambda: datetime(2026, 10, 28, tzinfo=UTC),
        )

    def safely(self, operation: Callable[[], Result]) -> Result:
        # Do not let unittest print raw provider/parser errors (which may contain credentials).
        failure_kind = ""
        result = None
        try:
            result = operation()
        except (APIError, RuntimeError, ValueError, OutputParserException) as error:
            failure_kind = type(error).__name__
        if failure_kind:
            self.fail(
                f"Live compatibility check failed ({failure_kind}); verify endpoint connectivity, credentials, and PARKING_EXTRACTION_METHOD. Raw errors suppressed."
            )
        assert result is not None
        return result

    def test_live_structured_extraction_support_and_verbatim_values(self):
        text = "Reserve parking. First name Demo, surname Tester, car number demo-bicycle-1."
        result = self.safely(
            lambda: self.agent._get_reservation_extractor().invoke(extraction_messages(ReservationSession(), text))
        )
        extraction = self.safely(lambda: parse_extraction(result))
        self.assertEqual(extraction.intent, "reservation")
        fields = {update.field: update.value for update in extraction.updates}
        self.assertEqual(fields, {"first_name": "Demo", "surname": "Tester", "car_number": "demo-bicycle-1"})

    def test_live_multiturn_collection_correction_and_code_confirmation(self):
        config: RunnableConfig = {"configurable": {"thread_id": "synthetic-smoke"}}
        graph = self.agent.graph
        for text in (
            "Reserve parking. First name Demo, surname Tester, car number demo-bicycle-1.",
            "I want to reserve for 2 hours.",
            "Change surname to Example.",
        ):
            result = self.safely(lambda text=text: graph.invoke({"messages": [("user", text)]}, config))
            if result["messages"][-1].content == EXTRACTION_FAILURE:
                self.fail(
                    "The endpoint returned a safe extraction failure; run the structured-extraction smoke check for the error category."
                )
        state = graph.get_state(config).values["reservation"]
        self.assertEqual(state["status"], "awaiting_confirmation")
        self.assertEqual(state["draft"]["surname"], "Example")
        self.assertEqual(state["request"]["duration_hours"], 2)
        self.assertNotIn("start", state["request"])
        self.assertNotIn("timezone", state["draft"])
        result = self.safely(lambda: graph.invoke({"messages": [("user", f"/confirm {state['revision']}")]}, config))
        self.assertTrue(result["messages"][-1].text.startswith(READY_REPLY))

    def test_live_budget_extraction_and_independent_python_validation(self):
        for text, minimum, maximum, inclusive in (
            ("Any available spaces under $8/hour?", None, 800, False),
            ("Show available spaces up to $8 per hour.", None, 800, True),
            ("I want a spot between $5 and $10/hour.", 500, 1000, True),
        ):
            with self.subTest(text=text):
                raw = self.safely(
                    lambda text=text: self.agent._get_reservation_extractor().invoke(
                        extraction_messages(ReservationSession(), text), config=guarded_model_config({})
                    )
                )
                proposal = self.safely(lambda raw=raw: parse_extraction(raw))
                self.assertIsNotNone(proposal.budget_phrase, "Model omitted the explicit budget phrase.")
                budget = self.safely(lambda text=text, proposal=proposal: resolve_budget(text, proposal.budget_phrase))
                assert budget is not None
                self.assertEqual(
                    (budget.minimum_cents, budget.maximum_cents, budget.maximum_inclusive),
                    (minimum, maximum, inclusive),
                )
                self.assertEqual(proposal.updates, [])

    def test_live_space_reference_is_not_a_vehicle_update(self):
        text = "Let's reserve OUT-P01 for 6 hours."
        raw = self.safely(
            lambda: self.agent._get_reservation_extractor().invoke(
                extraction_messages(ReservationSession(), text), config=guarded_model_config({})
            )
        )
        proposal = self.safely(lambda: parse_extraction(raw))
        fields = {update.field: update.value for update in proposal.updates}
        self.assertNotIn("car_number", fields)
        self.assertEqual(proposal.intent, "reservation")
        self.assertIn("duration_hours", fields)

    def test_live_negative_and_vague_corrections_are_structured_not_positive_values(self):
        for text, field in (("my vehicle id isn't OUT-P01", "car_number"), ("My surname is incorrect.", "surname")):
            with self.subTest(field=field):
                raw = self.safely(
                    lambda text=text: self.agent._get_reservation_extractor().invoke(
                        extraction_messages(ReservationSession(), text), config=guarded_model_config({})
                    )
                )
                proposal = self.safely(lambda raw=raw: parse_extraction(raw))
                self.assertEqual(proposal.updates, [])
                self.assertEqual([rejection.field for rejection in proposal.rejections], [field])
                self.assertIn(proposal.rejections[0].phrase, text)
        text = "Those details are wrong."
        raw = self.safely(
            lambda: self.agent._get_reservation_extractor().invoke(
                extraction_messages(ReservationSession(), text), config=guarded_model_config({})
            )
        )
        proposal = self.safely(lambda: parse_extraction(raw))
        self.assertIsNotNone(proposal.disagreement_phrase)
        self.assertEqual(proposal.updates, [])

    def check_live_administrator_review(self, *, asynchronous: bool):
        start = datetime.now(UTC)
        request = SubmittedReservationRequest(
            draft_id="synthetic-administrator-smoke",
            revision=1,
            location_id="demo-central",
            first_name="Demo",
            surname="Tester",
            car_number="demo-bicycle-1",
            duration_hours=1.5,
            start=start,
            end=start + timedelta(hours=1.5),
        )
        request_id = f"{request.draft_id}:{request.revision}"
        messages = AdministratorAgent._messages(request, "G-A01", request_id)
        config = guarded_model_config({})
        composer = self.safely(self.agent._get_administrator_composer)
        result = self.safely(
            lambda: (
                asyncio.run(composer.ainvoke(messages, config=config))
                if asynchronous
                else composer.invoke(messages, config=config)
            )
        )
        # Inspect parsed output BEFORE fallback handling: a visible review alone
        # does not prove that the real administrator model call succeeded.
        self.assertTrue(
            isinstance(result, dict) and result.get("parsing_error") is None,
            "Live administrator output could not be parsed; raw provider output suppressed.",
        )
        wording = self.safely(lambda: ReviewWording.model_validate(result.get("parsed")))
        payload = self.agent.administrator_agent._payload(request, "G-A01", request_id, result)
        self.assertTrue(
            bool(wording.message.strip()) and payload["message"] == wording.message.strip(),
            "Live administrator wording was rejected by the review guard; fallback would be used.",
        )
        self.assertEqual(payload["request"], request.model_dump(mode="json"))
        self.assertEqual(payload["request_id"], request_id)
        self.assertEqual(payload["space_id"], "G-A01")
        self.assertEqual(payload["status"], "pending")
        self.assertNotIn("approved", payload)

    def test_live_administrator_structured_output_and_canonical_payload(self):
        self.check_live_administrator_review(asynchronous=False)

    def test_live_async_administrator_structured_output_and_canonical_payload(self):
        self.check_live_administrator_review(asynchronous=True)
