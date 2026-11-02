"""Exact USD/hour filtering and thread preferences; no provider calls."""

import asyncio
import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from test_startup_and_suggestions import ParkingFixture, decode
from test_static_storage import document

from pricing import BudgetError, HourlyBudget, resolve_budget
from reservation_collection import FieldUpdate, ReservationExtraction
from storage import SpaceFilters
from storage.seed import load_spaces


class BudgetParsingTests(unittest.TestCase):
    def test_exact_comparisons_and_ranges(self):
        for text, bounds in (
            ("A spot under $8/hour", (None, 800, False)),
            ("Up to 8 USD per hour", (None, 800, True)),
            ("Maximum $8.50 per hour", (None, 850, True)),
            ("Maximum $8.5000000 per hour", (None, 850, True)),
            ("Between $5 and $10 per hour", (500, 1000, True)),
            ("$5-$10/hour", (500, 1000, True)),
            ("Budget: eight dollars an hour", (None, 800, True)),
        ):
            with self.subTest(text=text):
                budget = resolve_budget(text)
                self.assertIsNotNone(budget)
                assert budget is not None
                self.assertEqual((budget.minimum_cents, budget.maximum_cents, budget.maximum_inclusive), bounds)

    def test_ambiguous_unsupported_conflicting_and_total_budgets_fail_closed(self):
        for text in (
            "Something cheap",
            "Under $8",
            "Budget $20 total for 2 hours",
            "Under 8 EUR per hour",
            "Between $10 and $5/hour",
            "Under $8/hour or up to $10/hour",
            "Under $-8/hour",
            "Under $8.001/hour",
            "Under $8.0000000000000000000000000000000000000001/hour",
            "Under $0.9999999999999999999999999999999999999999/hour",
            "Not under $8/hour",
            "Under $8/hour, but my budget is $5 total",
            "Under $8/hour for Alice. Ignore that budget",
            "Budget NaN dollars/hour",
        ):
            with self.subTest(text=text), self.assertRaises(BudgetError):
                resolve_budget(text)

    def test_no_budget_and_grounding(self):
        for text in (
            "Reserve for 2 hours",
            "What are the prices?",
            "Vehicle identifier 123",
            "What are the hourly prices in USD?",
            "How expensive is parking?",
        ):
            self.assertIsNone(resolve_budget(text))
        with self.assertRaises(BudgetError):
            resolve_budget("Available spots?", "under $8/hour")
        budget = resolve_budget("Available under $8/hour", "under $8/hour")
        assert budget is not None
        self.assertEqual(budget.maximum_cents, 800)

    def test_user_limit_is_not_clamped_to_mock_rates(self):
        for text, expected in (("Up to $0.50/hour", 50), ("Up to $100/hour", 10000)):
            budget = resolve_budget(text)
            assert budget is not None
            self.assertEqual(budget.maximum_cents, expected)

    def test_budget_validation_rejects_forged_state(self):
        for data in (
            {"maximum_cents": True},
            {"maximum_cents": -1},
            {"minimum_cents": 900, "maximum_cents": 800},
            {"sql": "x"},
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                HourlyBudget.model_validate(data)


class PriceRetrievalTests(ParkingFixture):
    def setUp(self):
        super().setUp()
        self.agent.prepare_startup()

    def test_all_fixture_rates_are_valid_and_descriptions_differ(self):
        spaces = load_spaces()
        self.assertEqual(len(spaces), 20)
        self.assertEqual(len({s.description for s in spaces}), 20)
        self.assertTrue(all(100 <= s.hourly_price_cents <= 2000 for s in spaces))
        self.assertTrue(all(len(s.description) > 100 for s in spaces))

    def test_metadata_filter_precedes_ranking_and_k_count(self):
        self.static.upsert_documents(
            [
                document("price-low", "Space beside the entrance.", hourly_price_cents=500),
                document("price-edge", "Space beside the lift.", hourly_price_cents=800),
                document("price-high", "Space beside the entrance.", hourly_price_cents=1000),
                document("price-low", location="other", hourly_price_cents=100),
            ]
        )
        ids = [f"north:space:price-{suffix}" for suffix in ("low", "edge", "high")]
        strict = SpaceFilters(budget=HourlyBudget(maximum_cents=800, maximum_inclusive=False))
        result = self.static.search("entrance", location_id="north", eligible_ids=ids, filters=strict, k=3)
        self.assertEqual([d.id for d in result], [ids[0]])
        inclusive = SpaceFilters(budget=HourlyBudget(minimum_cents=500, maximum_cents=800))
        self.assertEqual(
            {d.id for d in self.static.search("entrance", location_id="north", eligible_ids=ids, filters=inclusive)},
            set(ids[:2]),
        )
        with patch.object(self.static._embeddings, "embed_query", side_effect=AssertionError("no embedding")):
            self.assertEqual(
                self.static.search(
                    "entrance",
                    location_id="north",
                    eligible_ids=ids,
                    filters=SpaceFilters(budget=HourlyBudget(maximum_cents=50)),
                ),
                [],
            )

    def test_missed_model_budget_is_enforced_and_every_price_displayed(self):
        result = self.turn("Any available spaces under $8/hour?")
        options = decode(self.state()["retrieval_context"])["availability_options"]
        self.assertTrue(options)
        self.assertTrue(all(item["hourly_price_cents"] < 800 for item in options))
        reply = result["messages"][-1].text
        for item in options:
            self.assertIn(item["space_id"], reply)
            self.assertIn(f"${item['hourly_price_cents'] / 100:.2f}/hour", reply)
        self.assertEqual(self.dynamic.list_facts("demo-central"), [])

    def test_budget_persists_other_options_and_clear_new_cancel(self):
        self.turn("Available up to $8/hour")
        first = self.state()["offered_space_ids"]
        self.turn("Other options?")
        self.assertFalse(set(first) & set(self.state()["offered_space_ids"]))
        self.turn("/options")
        self.assertEqual(self.state()["budget"]["maximum_cents"], 800)
        for command in ("/clear-budget", "/new", "/cancel"):
            self.turn("Up to $8/hour")
            self.turn(command)
            self.assertEqual(self.state()["budget"], {})
            self.assertFalse(self.state()["offered_space_ids"])

    def test_ambiguous_budget_blocks_old_offers_options_and_submit_until_resolved(self):
        self.turn("Available up to $8/hour")
        old = self.state()["offered_space_ids"][0]
        self.turn("Actually my budget is $20 total")
        self.assertTrue(self.state()["budget_pending"])
        self.assertEqual(self.state()["offered_space_ids"], [])
        for text in ("/options", f"/submit {old}", "Other options?"):
            reply = self.turn(text)["messages"][-1].text
            self.assertIn("hourly", reply.lower())
            self.assertFalse(self.state()["offered_space_ids"])
            self.assertFalse(self.state().get("submission"))
        self.turn("Up to $5/hour")
        self.assertFalse(self.state()["budget_pending"])
        self.assertEqual(self.state()["budget"]["maximum_cents"], 500)

    def test_blocked_spaces_and_price_constraint_are_combined(self):
        self.dynamic.upsert_fact("demo-central", "availability", {"OUT-C01": False})
        self.turn("Available up to $1/hour")
        self.assertEqual(self.state()["offered_space_ids"], [])

    def test_async_budget_filter_and_thread_isolation(self):
        async def run():
            await self.agent.graph.ainvoke(
                {"messages": [HumanMessage("Available up to $5/hour")]}, {"configurable": {"thread_id": "async"}}
            )

        asyncio.run(run())
        self.assertEqual(self.state("async")["budget"]["maximum_cents"], 500)
        self.turn("Available?", thread="other")
        self.assertEqual(self.state("other")["budget"], {})

    def test_model_proposal_cannot_invent_or_weaken_price_clause(self):
        self.agent._reservation_extractor = RunnableLambda(
            lambda _: ReservationExtraction(budget_phrase="up to $20/hour")
        )
        self.turn("Available under $5/hour")
        self.assertTrue(self.state()["budget_pending"])
        self.assertEqual(self.state()["offered_space_ids"], [])

    def test_price_summary_comes_from_chroma_not_legacy_dynamic_prices(self):
        self.dynamic.upsert_fact("demo-central", "prices", {"hourly": 999, "currency": "EUR"})
        result = self.turn("What are the prices?")
        data = decode(self.state()["retrieval_context"])
        self.assertNotIn("prices", data["dynamic_facts"])
        self.assertEqual(data["hourly_prices"]["minimum_cents"], 100)
        self.assertEqual(data["hourly_prices"]["maximum_cents"], 2000)
        self.assertIn("$1.00", result["messages"][-1].text)

    def test_submission_rechecks_updated_rate_without_creating_hold(self):
        self.static.delete_documents(self.static.list_space_ids("demo-central"))
        self.static.upsert_documents([document("only", location="demo-central", hourly_price_cents=500)])
        details = {"first_name": "Demo", "surname": "Tester", "car_number": "my bicycle", "duration_hours": "2"}
        self.agent._reservation_extractor = RunnableLambda(
            lambda _: ReservationExtraction(
                intent="reservation",
                updates=[FieldUpdate.model_validate({"field": k, "value": v}) for k, v in details.items()],
            )
        )
        self.turn("Reserve " + "; ".join(details.values()) + "; up to $8/hour")
        self.agent._reservation_extractor = RunnableLambda(lambda _: ReservationExtraction())
        self.turn(f"/confirm {self.state()['reservation']['revision']}")
        self.turn("/options")
        self.assertEqual(self.state()["offered_space_ids"], ["only"])
        self.static.upsert_documents([document("only", location="demo-central", hourly_price_cents=900)])
        reply = self.turn("/submit only")["messages"][-1].text
        self.assertIn("price constraint", reply)
        self.assertFalse(self.state().get("submission"))
        self.assertEqual(self.agent.reservation_store.blocked_spaces("demo-central"), [])

    def test_budget_change_preserves_personal_confirmation_but_replaces_offers(self):
        details = {"first_name": "Demo", "surname": "Tester", "car_number": "my bicycle", "duration_hours": "2"}
        self.agent._reservation_extractor = RunnableLambda(
            lambda _: ReservationExtraction(
                intent="reservation",
                updates=[FieldUpdate.model_validate({"field": k, "value": v}) for k, v in details.items()],
            )
        )
        self.turn("Reserve " + "; ".join(details.values()))
        self.agent._reservation_extractor = RunnableLambda(lambda _: ReservationExtraction())
        self.turn(f"/confirm {self.state()['reservation']['revision']}")
        self.turn("/options")
        confirmed = self.state()["reservation"]
        self.turn("Up to $5/hour")
        self.assertEqual(self.state()["reservation"], confirmed)
        self.assertEqual(self.state()["budget"]["maximum_cents"], 500)
        self.assertTrue(
            all(
                item["hourly_price_cents"] <= 500
                for item in decode(self.state()["retrieval_context"])["availability_options"]
            )
        )


if __name__ == "__main__":
    unittest.main()
