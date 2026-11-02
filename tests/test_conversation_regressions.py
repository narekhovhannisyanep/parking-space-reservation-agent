"""Screenshot regressions: injected bad extraction, real isolated ledger, no provider."""

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from test_recording_mcp import in_process_recorder

from administrator_agent import ReviewWording
from guardrails import PrivacyGuard
from parking_agent import AgentConfig, ParkingSpaceReservationAgent
from reservation_collection import FieldRejection, FieldUpdate, ReservationExtraction, ValidatedReservationRequest
from storage import DynamicParkingStore, ParkingSpace, StaticParkingStore, StorageConfig


def extraction(**fields):
    return ReservationExtraction(
        intent="reservation",
        updates=[FieldUpdate.model_validate({"field": name, "value": value}) for name, value in fields.items()],
    )


class ConversationRegressionTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.storage = StorageConfig(chroma_directory=root / "chroma", sqlite_path=root / "parking.sqlite3")
        self.now = datetime(2030, 1, 1, tzinfo=UTC)
        self.guard = PrivacyGuard(lambda _: [])
        self.dynamic = DynamicParkingStore(self.storage, privacy_guard=self.guard)
        self.documents = {
            space.document_id: space.to_document()
            for space in (
                ParkingSpace(
                    space_id=identifier,
                    location_id="demo-central",
                    label="Demo space",
                    description=description,
                    source="synthetic-test",
                    hourly_price_cents=price,
                )
                for identifier, description, price in (
                    ("OUT-P01", "Convenient pedestrian route for shopping bags.", 500),
                    ("G-E02", "Beside the lift with a sheltered route.", 700),
                    ("G-A03", "Near the entrance with an unloading route.", 800),
                    ("L1-B01", "Beside stairs on the first level.", 1000),
                    ("OUT-C01", "Along the outdoor fence.", 100),
                )
            )
        }
        self.static = Mock(spec=StaticParkingStore)
        self.static.list_space_ids.side_effect = lambda _: list(self.documents)
        self.static.get_spaces.side_effect = lambda ids, **_: [self.documents[identifier] for identifier in ids]

        def search(query, *, eligible_ids=None, k=3, filters=None, **_):
            candidates = [self.documents[identifier] for identifier in eligible_ids or []]
            if filters is not None and filters.budget is not None:
                candidates = [doc for doc in candidates if filters.budget.accepts(doc.metadata["hourly_price_cents"])]
            # Make losing the original preference visibly change the selected set.
            if "shopping" not in query:
                candidates.reverse()
            return candidates[:k]

        self.static.search.side_effect = search
        self.static.hourly_price_summary.return_value = {
            "status": "inventory",
            "currency": "USD",
            "minimum_cents": 100,
            "maximum_cents": 1000,
        }
        self.proposal = ReservationExtraction()
        self.model = Mock()
        self.model.invoke.return_value = AIMessage("Parking information.")
        self.model.ainvoke = AsyncMock(return_value=AIMessage("Parking information."))
        self.agent = ParkingSpaceReservationAgent(
            config=AgentConfig(storage=self.storage),
            static_store=self.static,
            dynamic_store=self.dynamic,
            privacy_guard=self.guard,
            llm_client=self.model,
            reservation_extractor=RunnableLambda(lambda _: self.proposal),
            administrator_composer=RunnableLambda(lambda _: ReviewWording(message="Please review this request.")),
            recording_client=in_process_recorder(self.storage),
            checkpointer=InMemorySaver(),
            clock=lambda: self.now,
        )
        self.agent.recording_writer.reset_output_directory()
        self.config: RunnableConfig = {"configurable": {"thread_id": "screenshot"}}

    def turn(self, text, proposal=None):
        self.proposal = proposal or ReservationExtraction()
        return self.agent.graph.invoke({"messages": [HumanMessage(text)]}, self.config)

    def state(self):
        return self.agent.graph.get_state(self.config).values

    def fill(self, vehicle="demo-bicycle-1"):
        self.turn(
            f"First name Demo, surname Tester, vehicle identifier {vehicle}, reserve for 6 hours.",
            extraction(first_name="Demo", surname="Tester", car_number=vehicle, duration_hours="6 hours"),
        )

    def confirm(self):
        return self.turn(f"/confirm {self.state()['reservation']['revision']}")

    def test_unrelated_detour_preserves_confirmed_options_and_budget(self):
        self.turn("Parking spaces for shopping bags up to $8/hour.")
        self.fill()
        self.confirm()
        before = self.state()
        self.static.reset_mock()
        reply = self.turn("What are the restaurant prices?", ReservationExtraction(intent="unrelated"))["messages"][
            -1
        ].text
        after = self.state()
        for key in (
            "reservation",
            "budget",
            "offered_space_ids",
            "offer_candidate_ids",
            "offer_query",
            "offer_round_ids",
            "offer_duration_hours",
        ):
            self.assertEqual(after[key], before[key])
        self.assertEqual(after["retrieval_context"], "")
        self.assertNotIn("Current parking options", reply)
        self.static.search.assert_not_called()
        self.turn("/options")
        self.assertEqual(self.state()["offered_space_ids"], before["offered_space_ids"])

    def test_unrelated_cost_cue_does_not_modify_pending_parking_budget(self):
        self.turn("Parking under about $8/hour.")
        self.assertTrue(self.state()["budget_pending"])
        self.turn("Restaurant meals under $20", ReservationExtraction(intent="unrelated"))
        self.assertTrue(self.state()["budget_pending"])
        self.assertEqual(self.state()["retrieval_context"], "")
        self.assertEqual(self.model.invoke.call_args.args[0][-1].text, "Restaurant meals under $20")

    def test_space_selection_cannot_become_vehicle_even_with_bad_model_output(self):
        self.turn("Any parking spaces for shopping bags?")
        first = self.state()["offered_space_ids"][:]
        reply = self.turn(
            "Let's reserve OUT-P01 for 6 hours.", extraction(car_number="OUT-P01", duration_hours="6 hours")
        )["messages"][-1].text
        self.assertIsNone(self.state()["reservation"]["draft"]["car_number"])
        self.assertEqual(self.state()["reservation"]["draft"]["duration_hours"], "6 hours")
        self.assertIn("vehicle identifier", reply)
        self.assertEqual(self.state()["offered_space_ids"], first)
        self.assertEqual(self.static.search.call_count, 1)
        self.assertFalse(self.state().get("submission"))

    def test_space_id_level_prefix_is_not_a_new_descriptive_preference(self):
        self.turn("Available spaces?")
        first = self.state()["offered_space_ids"][:]
        self.assertIn("L1-B01", first)
        self.turn("Let's reserve L1-B01 for 6 hours.", extraction(duration_hours="6 hours"))
        self.assertEqual(self.state()["offered_space_ids"], first)
        self.assertEqual(self.static.search.call_count, 1)
        self.assertEqual(self.state()["offer_query"], "availability")

    def test_fragment_of_an_offered_space_id_cannot_become_a_vehicle_identifier(self):
        self.turn("Any parking spaces for shopping bags?")
        self.turn("Let's reserve OUT-P01 for 6 hours.", extraction(car_number="P01", duration_hours="6 hours"))
        self.assertIsNone(self.state()["reservation"]["draft"]["car_number"])

    def test_reversed_negative_statement_clears_the_rejected_vehicle(self):
        self.fill()
        self.confirm()
        self.turn("demo-bicycle-1 is not my vehicle identifier.", extraction(car_number="demo-bicycle-1"))
        self.assertIsNone(self.state()["reservation"]["draft"]["car_number"])
        self.assertIsNone(self.state()["reservation"]["confirmed_revision"])

    def test_explicit_vehicle_may_legitimately_equal_a_space_id(self):
        self.fill("OUT-P01")
        self.assertEqual(self.state()["reservation"]["draft"]["car_number"], "OUT-P01")
        self.assertEqual(self.state()["reservation"]["status"], "awaiting_confirmation")

    def test_negative_vehicle_correction_clears_field_even_if_model_repeats_it(self):
        self.fill("OUT-P01")
        self.confirm()
        old = self.state()["reservation"]["revision"]
        reply = self.turn("my vehicle id isn't OUT-P01", extraction(car_number="OUT-P01"))["messages"][-1].text
        session = self.state()["reservation"]
        self.assertIsNone(session["draft"]["car_number"])
        self.assertEqual(session["draft"]["first_name"], "Demo")
        self.assertIsNone(session["request"])
        self.assertIsNone(session["confirmed_revision"])
        self.assertGreater(session["revision"], old)
        self.assertNotIn("Please confirm", reply)
        self.turn(f"/confirm {old}")
        self.assertNotEqual(self.state()["reservation"]["status"], "ready")

    def test_rejection_with_replacement_uses_only_positive_occurrence(self):
        self.fill("OUT-P01")
        self.turn("My vehicle identifier isn't OUT-P01; it's demo-bicycle-2.", extraction(car_number="demo-bicycle-2"))
        self.assertEqual(self.state()["reservation"]["draft"]["car_number"], "demo-bicycle-2")
        self.assertEqual(self.state()["reservation"]["status"], "awaiting_confirmation")

    def test_rejection_and_replacement_without_punctuation_are_separate_clauses(self):
        self.fill("OUT-P01")
        self.turn("My vehicle id isn't OUT-P01 it's demo-bicycle-2", extraction(car_number="demo-bicycle-2"))
        self.assertEqual(self.state()["reservation"]["draft"]["car_number"], "demo-bicycle-2")

    def test_negated_full_name_does_not_partially_update_surname(self):
        self.fill()
        self.turn("I am not Alice Smith.", extraction(first_name="Alice", surname="Smith"))
        self.assertEqual(self.state()["reservation"]["draft"]["first_name"], "Demo")
        self.assertEqual(self.state()["reservation"]["draft"]["surname"], "Tester")

    def test_vague_contracted_disagreement_and_information_detour(self):
        self.fill()
        self.confirm()
        reply = self.turn("My info isn't correct.")["messages"][-1].text
        self.assertIn("which", reply)
        self.turn("yes")
        self.assertNotEqual(self.state()["reservation"]["status"], "ready")
        reply = self.turn("What are the prices?")["messages"][-1].text
        self.assertIn("Inventory hourly prices", reply)
        self.assertTrue(self.state()["reservation"]["correction_pending"])

    def test_named_rejections_work_without_a_model_patch_for_all_fields(self):
        for field, text in (
            ("first_name", "My first name is incorrect."),
            ("surname", "My surname isn't Tester."),
            ("car_number", "The car number is wrong."),
            ("duration_hours", "The duration is incorrect."),
        ):
            with self.subTest(field=field):
                self.turn("/new")
                self.fill()
                reply = self.turn(text)["messages"][-1].text
                self.assertIsNone(self.state()["reservation"]["draft"][field])
                self.assertNotIn("Please confirm", reply)

    def test_vague_disagreement_blocks_yes_and_submit_until_specific_correction(self):
        self.fill()
        self.confirm()
        before = self.state()["reservation"]["draft"]
        reply = self.turn("Those details are wrong.")["messages"][-1].text
        self.assertIn("which", reply.lower())
        self.assertEqual(self.state()["reservation"]["draft"], before)
        self.assertIsNone(self.state()["reservation"]["confirmed_revision"])
        self.turn("yes")
        self.assertNotEqual(self.state()["reservation"]["status"], "ready")
        self.turn("/submit OUT-P01")
        self.assertFalse(self.state().get("submission"))
        self.turn("My surname is Example.", extraction(surname="Example"))
        self.confirm()
        self.assertEqual(self.state()["reservation"]["status"], "ready")

    def test_negative_and_positive_same_value_are_distinguished_by_occurrence(self):
        self.fill()
        self.turn("My surname is not Demo; my surname is Tester.", extraction(surname="Tester"))
        self.assertEqual(self.state()["reservation"]["draft"]["surname"], "Tester")
        self.turn("My surname is not Tester.", extraction(surname="Tester"))
        self.assertIsNone(self.state()["reservation"]["draft"]["surname"])

    def test_confirmation_retains_browse_options_and_submits_without_options_command(self):
        self.turn("Any parking spaces for shopping bags?")
        first = self.state()["offered_space_ids"][:]
        self.fill()
        reply = self.confirm()["messages"][-1].text
        self.assertEqual(self.state()["offered_space_ids"], first)
        self.assertEqual(self.state()["offer_duration_hours"], 6)
        self.assertEqual(self.static.search.call_count, 1)
        self.assertNotIn("Send /options", reply)
        result = self.turn("/submit OUT-P01")
        pending = result["__interrupt__"][0]
        self.assertEqual(pending.value["space_id"], "OUT-P01")
        self.assertEqual(pending.value["request"]["car_number"], "demo-bicycle-1")
        result = self.agent.graph.invoke(Command(resume={pending.id: {"decision": "approve"}}), self.config)
        self.assertIn("Recorded successfully", result["messages"][-1].text)
        self.assertEqual(len(list(self.agent.config.recording.directory.glob("*.txt"))), 1)

    def test_confirmation_without_browsing_automatically_offers_options(self):
        self.fill()
        reply = self.confirm()["messages"][-1].text
        self.assertEqual(len(self.state()["offered_space_ids"]), 3)
        self.assertIn("Current parking options:", reply)
        self.assertIn("/submit SPACE_ID", reply)
        self.assertFalse(self.state().get("submission"))

    def test_options_rechecks_same_set_other_options_requests_a_different_set(self):
        self.turn("Any parking spaces for shopping bags?")
        first = self.state()["offered_space_ids"][:]
        self.turn("/options")
        self.assertEqual(self.state()["offered_space_ids"], first)
        self.assertEqual(self.static.search.call_count, 1)
        self.turn("Other options?")
        self.assertFalse(set(first) & set(self.state()["offered_space_ids"]))
        self.assertIn("shopping", self.static.search.call_args.args[0])

    def test_revalidation_explains_unavailable_options_and_keeps_survivor_order(self):
        self.turn("Any parking spaces for shopping bags?")
        first = self.state()["offered_space_ids"][:]
        self.fill()
        self.dynamic.upsert_fact("demo-central", "availability", {first[0]: False})
        reply = self.confirm()["messages"][-1].text
        self.assertEqual(self.state()["offered_space_ids"][:2], first[1:])
        self.assertNotIn(first[0], self.state()["offered_space_ids"])
        self.assertIn("no longer eligible", reply)
        self.assertIn(first[0], reply)
        self.assertIn("shopping", self.static.search.call_args.args[0])

    def test_duration_change_rechecks_overlap_without_discarding_preferences(self):
        self.turn("Any parking spaces for shopping bags?")
        self.fill()
        self.confirm()
        first = self.state()["offered_space_ids"][:]
        # A future hold does not overlap a six-hour request, but overlaps eight hours.
        request = ValidatedReservationRequest.model_validate(self.state()["reservation"]["request"])
        self.now += timedelta(hours=7)
        self.agent.reservation_store.hold(
            request.model_copy(update={"draft_id": "competitor"}), first[0], "other", list(self.documents)
        )
        self.now -= timedelta(hours=7)
        self.turn("Change duration to 8 hours.", extraction(duration_hours="8 hours"))
        self.assertIsNone(self.state()["offer_duration_hours"])
        self.turn(f"/submit {first[0]}")
        self.assertFalse(self.state().get("submission"))
        self.confirm()
        self.assertNotIn(first[0], self.state()["offered_space_ids"])
        self.assertEqual(self.state()["offered_space_ids"][:2], first[1:])

    def test_correcting_names_does_not_reset_options_and_new_clears_context(self):
        self.turn("Any parking spaces for shopping bags?")
        self.fill()
        self.confirm()
        first = self.state()["offered_space_ids"][:]
        self.turn("My surname is wrong.")
        self.turn("My surname is Example.", extraction(surname="Example"))
        self.confirm()
        self.assertEqual(self.state()["offered_space_ids"], first)
        self.turn("/new")
        self.assertFalse(self.state()["offered_space_ids"])
        self.assertFalse(self.state().get("offer_candidate_ids"))
        self.assertFalse(self.state().get("offer_query"))

    def test_async_negative_correction_invalidates_ready_state(self):
        self.fill("OUT-P01")
        self.confirm()
        self.proposal = extraction(car_number="OUT-P01")
        asyncio.run(self.agent.graph.ainvoke({"messages": [HumanMessage("my vehicle id isn't OUT-P01")]}, self.config))
        self.assertIsNone(self.state()["reservation"]["draft"]["car_number"])
        self.assertIsNone(self.state()["reservation"]["confirmed_revision"])

    def test_forged_rejection_cannot_clear_unmentioned_fields_or_confirm(self):
        self.fill()
        self.confirm()
        for phrase in ("My surname is wrong.", "The entrance is wrong."):
            with self.subTest(phrase=phrase):
                proposal = ReservationExtraction(
                    intent="reservation", rejections=[FieldRejection(field="car_number", phrase=phrase)]
                )
                self.turn("The entrance is wrong.", proposal)
                self.assertEqual(self.state()["reservation"]["draft"]["car_number"], "demo-bicycle-1")
                self.assertIsNone(self.state()["reservation"]["confirmed_revision"])

    def test_failed_revalidation_cannot_leave_submittable_stale_options(self):
        self.turn("Any parking spaces for shopping bags?")
        self.fill()
        self.dynamic.list_facts = Mock(side_effect=OSError("private storage path"))
        with self.assertRaisesRegex(RuntimeError, "Unable to retrieve"):
            self.confirm()
        self.assertEqual(self.state()["reservation"]["status"], "ready")
        self.assertFalse(self.state()["offered_space_ids"])
        self.assertIsNone(self.state()["offer_duration_hours"])
        self.turn("/submit OUT-P01")
        self.assertFalse(self.state().get("submission"))

    def test_revalidation_reads_current_price_without_over_budget_padding(self):
        self.turn("Parking spaces for shopping bags up to $8/hour.")
        first = self.state()["offered_space_ids"][:]
        self.fill()
        identifier = f"demo-central:space:{first[0]}"
        self.documents[identifier].metadata["hourly_price_cents"] = 900
        reply = self.confirm()["messages"][-1].text
        self.assertNotIn(first[0], self.state()["offered_space_ids"])
        self.assertEqual(self.state()["offered_space_ids"][:2], first[1:])
        self.assertIn("no longer eligible", reply)
        self.assertTrue(
            all(
                self.documents[f"demo-central:space:{space}"].metadata["hourly_price_cents"] <= 800
                for space in self.state()["offered_space_ids"]
            )
        )

    def test_general_information_during_disagreement_does_not_confirm_details(self):
        self.fill()
        self.turn("Those details are wrong.")
        self.turn("What are the prices?")
        self.assertIsNone(self.state()["reservation"]["confirmed_revision"])
        self.assertTrue(self.state()["reservation"]["correction_pending"])

    def test_revalidation_does_not_resurrect_excluded_earlier_alternatives(self):
        self.turn("Any parking spaces for shopping bags?")
        initial = self.state()["offered_space_ids"][:]
        self.turn("Other options?")
        other = self.state()["offered_space_ids"][:]
        self.fill()
        self.confirm()
        self.assertEqual(self.state()["offered_space_ids"], other)
        self.assertFalse(set(initial) & set(self.state()["offered_space_ids"]))

    def test_other_options_exclude_temporarily_removed_browse_candidates(self):
        self.turn("Any parking spaces for shopping bags?")
        initial = self.state()["offered_space_ids"][:]
        self.fill()
        self.dynamic.upsert_fact("demo-central", "availability", {initial[0]: False})
        self.confirm()
        self.dynamic.delete_fact("demo-central", "availability")
        self.turn("Other options?")
        self.assertFalse(set(initial) & set(self.state()["offered_space_ids"]))

    def test_async_confirmation_retains_options_and_does_not_cross_threads(self):
        self.turn("Any parking spaces for shopping bags?")
        first = self.state()["offered_space_ids"][:]
        self.fill()
        self.proposal = ReservationExtraction()
        asyncio.run(self.agent.graph.ainvoke({"messages": [HumanMessage("yes")]}, self.config))
        self.assertEqual(self.state()["offered_space_ids"], first)
        other: RunnableConfig = {"configurable": {"thread_id": "another"}}
        asyncio.run(self.agent.graph.ainvoke({"messages": [HumanMessage("Available spaces?")]}, other))
        self.assertNotEqual(self.agent.graph.get_state(other).values["offer_query"], self.state()["offer_query"])
        self.assertEqual(self.state()["offered_space_ids"], first)
