"""Opt-in endpoint compatibility smoke checks using synthetic details and mocked public stores."""

import os
import unittest
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import TypeVar
from unittest.mock import Mock

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from openai import APIError

from parking_agent import AgentConfig, ParkingSpaceReservationAgent
from reservation_collection import (
    EXTRACTION_FAILURE,
    READY_REPLY,
    ReservationSession,
    extraction_messages,
    parse_extraction,
)
from storage import DynamicParkingStore, StaticParkingStore

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
        dynamic = Mock(spec=DynamicParkingStore)
        dynamic.list_facts.return_value = []
        self.agent = ParkingSpaceReservationAgent(
            config=replace(AgentConfig.from_environment(), timeout=15, max_retries=0, reasoning_effort="low"),
            static_store=static,
            dynamic_store=dynamic,
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
            "Start 2030-06-01 10:00, end 2030-06-01 12:00, timezone UTC+04:00.",
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
        # Compare instants, not Pydantic's UTC serialization ("Z" versus "+00:00").
        self.assertEqual(datetime.fromisoformat(state["request"]["start"]), datetime(2030, 6, 1, 6, tzinfo=UTC))
        result = self.safely(lambda: graph.invoke({"messages": [("user", f"/confirm {state['revision']}")]}, config))
        self.assertEqual(result["messages"][-1].content, READY_REPLY)
