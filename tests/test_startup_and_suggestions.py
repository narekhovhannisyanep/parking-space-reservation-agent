"""Server-startup and eligibility-constrained semantic retrieval checks using offline stores."""

import asyncio
import json
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, current_thread
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from starlette.testclient import TestClient
from test_static_storage import KeywordEmbeddings, document

import app
from context_augmentation import CONTEXT_HEADER, ContextConfig, offered_space_ids, retrieve_context
from parking_agent import AgentConfig, ParkingSpaceReservationAgent, ParkingState
from reservation_collection import ReservationExtraction
from storage import DynamicParkingStore, StaticParkingStore, StorageConfig
from storage.dynamic import JSONValue
from storage.seed import load_documents


def decode(context):
    return json.loads(context.removeprefix(CONTEXT_HEADER))


class ParkingFixture(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.storage = StorageConfig(chroma_directory=self.root / "chroma", sqlite_path=self.root / "parking.sqlite3")
        self.static = StaticParkingStore(self.storage, embeddings=KeywordEmbeddings())
        self.dynamic = DynamicParkingStore(self.storage)
        self.model = Mock()
        self.model.invoke.side_effect = lambda *args, **kwargs: AIMessage("Parking information")
        self.model.ainvoke = AsyncMock(side_effect=lambda *args, **kwargs: AIMessage("Parking information"))
        self.agent = ParkingSpaceReservationAgent(
            config=AgentConfig(storage=self.storage),
            static_store=self.static,
            dynamic_store=self.dynamic,
            llm_client=self.model,
            reservation_extractor=RunnableLambda(lambda _: ReservationExtraction()),
            checkpointer=InMemorySaver(),
            require_startup=True,
        )

    def turn(self, text="Available spaces?", thread="alice"):
        return self.agent.graph.invoke({"messages": [HumanMessage(text)]}, {"configurable": {"thread_id": thread}})

    def state(self, thread="alice"):
        return self.agent.graph.get_state({"configurable": {"thread_id": thread}}).values


class StartupTests(ParkingFixture):
    def test_first_startup_seeds_static_resets_dynamic_and_prepares_context(self):
        self.dynamic.upsert_fact("demo-central", "prices", {"hourly": 999})
        self.dynamic.upsert_fact("other", "availability", False)
        self.agent.prepare_startup()
        self.assertTrue(self.agent.startup_ready)
        self.assertEqual(len(self.static.list_space_ids("demo-central")), 20)
        self.assertEqual(self.dynamic.list_facts("demo-central"), [])
        self.assertEqual(self.dynamic.list_facts("other"), [])
        evidence = decode(self.agent.baseline_context)
        self.assertEqual(len(evidence["static_spaces"]), 3)
        self.assertNotIn("availability_options", evidence)
        self.assertEqual(evidence["dynamic_facts"]["prices"], {"status": "missing"})
        self.model.invoke.assert_not_called()

    def test_populated_static_is_reused_without_loading_seed_or_embeddings(self):
        original = document("custom", location="demo-central", text="Operator-maintained directions.")
        assert original.id is not None
        self.static.upsert_documents([original])
        with (
            patch("storage.seed.load_documents", side_effect=AssertionError("Do not reload fixture")),
            patch.object(self.static._embeddings, "embed_documents", side_effect=AssertionError("Do not reembed")),
            patch.object(self.static._embeddings, "embed_query", side_effect=AssertionError("No query needed")),
        ):
            self.agent.prepare_startup()
        self.assertEqual(self.static.get_spaces([original.id], location_id="demo-central"), [original])
        self.assertEqual(len(self.static.list_space_ids("demo-central")), 1)

    def test_existing_empty_collection_is_seeded_immediately(self):
        self.static._get_store()
        self.assertEqual(self.static.list_space_ids("demo-central"), [])
        with patch("storage.seed.load_documents", wraps=load_documents) as load:
            self.static.initialize()
        load.assert_called_once_with()
        self.assertEqual(len(self.static.list_space_ids("demo-central")), 20)
        self.assertFalse(self.storage.sqlite_path.exists())

    def test_restart_reuses_static_and_resets_sqlite_again(self):
        self.agent.prepare_startup()
        ids = self.static.list_space_ids("demo-central")
        before = self.static.get_spaces(ids, location_id="demo-central")
        self.dynamic.upsert_fact("demo-central", "availability", False)
        self.agent.finish_shutdown()
        self.agent.prepare_startup()
        self.assertEqual(self.dynamic.list_facts("demo-central"), [])
        self.assertEqual(self.static.get_spaces(ids, location_id="demo-central"), before)

    def test_repeated_and_concurrent_preparation_does_not_reset_active_data(self):
        with patch.object(self.dynamic, "reset", wraps=self.dynamic.reset) as reset:
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(lambda _: self.agent.prepare_startup(), range(16)))
            self.dynamic.upsert_fact("demo-central", "prices", 5)
            self.agent.prepare_startup()
        reset.assert_called_once_with()
        fact = self.dynamic.get_fact("demo-central", "prices")
        assert fact is not None
        self.assertEqual(fact.value, 5)

    def test_static_failure_stops_startup_without_resetting_sqlite(self):
        with (
            patch.object(self.static, "initialize", side_effect=OSError("private/path")),
            patch.object(self.dynamic, "reset") as reset,
            self.assertRaisesRegex(RuntimeError, "static database initialization/seeding") as error,
        ):
            self.agent.prepare_startup()
        self.assertFalse(self.agent.startup_ready)
        self.assertNotIn("private", str(error.exception))
        reset.assert_not_called()
        self.agent.prepare_startup()
        self.assertTrue(self.agent.startup_ready)

    def test_seed_failure_propagates_and_retry_can_seed_empty_collection(self):
        with (
            patch("storage.seed.load_documents", side_effect=ValueError("bad fixture")),
            self.assertRaisesRegex(RuntimeError, "initialization/seeding"),
        ):
            self.agent.prepare_startup()
        self.assertFalse(self.agent.startup_ready)
        self.agent.prepare_startup()
        self.assertEqual(len(self.static.list_space_ids("demo-central")), 20)

    def test_dynamic_reset_and_read_failures_stop_startup(self):
        for method, stage in (("reset", "dynamic database reset"), ("list_facts", "baseline context retrieval")):
            with self.subTest(method=method):
                with (
                    patch.object(self.dynamic, method, side_effect=sqlite3.OperationalError("private SQL")),
                    self.assertRaisesRegex(RuntimeError, stage),
                ):
                    self.agent.prepare_startup()
                self.assertFalse(self.agent.startup_ready)
                self.assertEqual(self.agent.baseline_context, "")

    def test_missing_location_fails_without_overwriting_populated_collection(self):
        self.static.upsert_documents([document(location="other")])
        with self.assertRaisesRegex(RuntimeError, "baseline context retrieval"):
            self.agent.prepare_startup()
        self.assertFalse(self.agent.startup_ready)
        self.assertEqual(len(self.static.list_space_ids("other")), 1)

    def test_unusable_baseline_fails_and_budget_is_never_exceeded(self):
        huge = document(location="demo-central")
        huge.page_content = "x" * 5000
        self.static.upsert_documents([huge])
        self.agent.config = replace(self.agent.config, context=ContextConfig(max_context_chars=2000))
        with self.assertRaisesRegex(RuntimeError, "baseline context retrieval"):
            self.agent.prepare_startup()
        self.assertFalse(self.agent.startup_ready)

    def test_unprepared_server_graph_rejects_messages_before_any_model_call(self):
        with self.assertRaisesRegex(RuntimeError, "databases are not ready"):
            self.turn("/new")
        self.model.invoke.assert_not_called()
        self.assertFalse(self.storage.chroma_directory.exists())
        self.assertFalse(self.storage.sqlite_path.exists())

    def test_startup_does_not_implement_recording_output_cleanup(self):
        output = self.root / "approved-reservations"
        output.mkdir()
        existing = output / "existing.txt"
        existing.write_text("Stage 3 output is untouched in this change.")
        self.agent.prepare_startup()
        self.assertTrue(existing.exists())


class SuggestionTests(ParkingFixture):
    def setUp(self):
        super().setUp()
        self.agent.prepare_startup()

    def test_user_topic_query_changes_similarity_ranking_and_never_exceeds_three(self):
        self.agent.config = replace(self.agent.config, context=ContextConfig(retrieval_k=20))
        with patch.object(self.static._embeddings, "embed_query", wraps=self.static._embeddings.embed_query) as embed:
            for message, expected_query, expected_space in (
                ("Any available spaces near the elevator?", "lift availability", "G-E02"),
                ("Any available spaces near the entrance?", "entrance availability", "G-A01"),
            ):
                self.turn(message)
                ids = self.state()["offered_space_ids"]
                self.assertEqual(len(ids), 3)
                self.assertEqual(len(set(ids)), 3)
                self.assertEqual(ids[0], expected_space)
                self.assertEqual(embed.call_args.args, (expected_query,))
        self.assertEqual(embed.call_count, 2)

    def test_three_distinct_semantic_options_are_scoped_without_writes(self):
        self.static.upsert_documents([document(location="other")])
        with (
            patch.object(self.static, "upsert_documents") as static_write,
            patch.object(self.dynamic, "upsert_fact") as dynamic_write,
            patch.object(self.static, "search", wraps=self.static.search) as search,
        ):
            result = self.turn()
        evidence = decode(self.state()["retrieval_context"])
        options = evidence["availability_options"]
        self.assertEqual(len(options), 3)
        self.assertEqual(len({item["space_id"] for item in options}), 3)
        self.assertEqual(self.state()["offered_space_ids"], [item["space_id"] for item in options])
        self.assertEqual(search.call_args.args, ("availability",))
        self.assertEqual(search.call_args.kwargs["k"], 3)
        self.assertTrue(
            all(identifier.startswith("demo-central:space:") for identifier in search.call_args.kwargs["eligible_ids"])
        )
        self.assertEqual(evidence["static_spaces"], [])
        self.assertEqual(result["messages"][0].content, "Available spaces?")
        static_write.assert_not_called()
        dynamic_write.assert_not_called()
        self.assertEqual(self.dynamic.list_facts("demo-central"), [])

    def test_dynamic_overrides_filter_before_ranking_and_reread_each_turn(self):
        ids = self.static.list_space_ids("demo-central")
        blocked: dict[str, JSONValue] = {
            identifier.removeprefix("demo-central:space:"): False for identifier in ids[:-2]
        }
        self.dynamic.upsert_fact("demo-central", "availability", blocked)
        self.turn()
        self.assertEqual(set(self.state()["offered_space_ids"]), {ids[-2].split(":")[-1], ids[-1].split(":")[-1]})
        self.dynamic.upsert_fact("demo-central", "availability", False)
        self.turn()
        self.assertEqual(self.state()["offered_space_ids"], [])
        self.dynamic.delete_fact("demo-central", "availability")
        self.turn()
        self.assertEqual(len(self.state()["offered_space_ids"]), 3)

    def test_blocked_best_match_is_excluded_before_semantic_ranking(self):
        self.dynamic.upsert_fact("demo-central", "availability", {"G-E02": False})
        self.turn("Available spaces near the elevator?")
        ids = self.state()["offered_space_ids"]
        self.assertEqual(len(ids), 3)
        self.assertNotIn("G-E02", ids)

    def test_private_message_details_are_not_forwarded_to_embeddings(self):
        text = "Alice Smith ABC-123 wants available spaces near the elevator for 2030-06-01 10:00."
        with patch.object(self.static._embeddings, "embed_query", wraps=self.static._embeddings.embed_query) as embed:
            self.turn(text)
        embed.assert_called_once_with("lift availability")
        self.assertEqual(self.state()["messages"][0].content, text)

    def test_empty_eligibility_does_not_embed_and_embedding_failure_stops_response(self):
        self.dynamic.upsert_fact("demo-central", "availability", False)
        with patch.object(
            self.static._embeddings, "embed_query", side_effect=RuntimeError("Embedding unavailable")
        ) as embed:
            self.turn()
            embed.assert_not_called()
            self.dynamic.delete_fact("demo-central", "availability")
            self.model.invoke.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "Embedding unavailable"):
                self.turn()
        self.model.invoke.assert_not_called()

    def test_fewer_than_three_empty_inventory_and_global_true_are_supported(self):
        ids = self.static.list_space_ids("demo-central")
        self.static.delete_documents(ids[1:])
        self.dynamic.upsert_fact("demo-central", "availability", True)
        self.turn()
        self.assertEqual(len(self.state()["offered_space_ids"]), 1)
        self.static.delete_documents(ids[:1])
        self.turn()
        self.assertEqual(self.state()["offered_space_ids"], [])

    def test_malformed_or_null_availability_cannot_silently_offer_spaces(self):
        values: list[JSONValue] = [None, {"free": 3}, {"G-A01": "false"}, ["G-A01"], {"../bad": False}]
        for value in values:
            with self.subTest(value=value):
                self.dynamic.upsert_fact("demo-central", "availability", value)
                self.model.invoke.reset_mock()
                with self.assertRaises((RuntimeError, ValueError)):
                    self.turn()
                self.model.invoke.assert_not_called()

    def test_failed_read_is_not_empty_and_prevents_model_response(self):
        with (
            patch.object(self.dynamic, "list_facts", side_effect=sqlite3.OperationalError("private")),
            self.assertRaisesRegex(RuntimeError, "Unable to retrieve"),
        ):
            self.turn()
        self.model.invoke.assert_not_called()

    def test_hours_and_prices_do_not_retrieve_or_offer_space_inventory(self):
        self.dynamic.upsert_fact("demo-central", "prices", {"hourly": 9})
        self.dynamic.upsert_fact("demo-central", "availability", {"G-A01": False})
        with patch.object(self.static, "search") as search, patch.object(self.static, "list_space_ids") as ids:
            self.turn("Prices and working hours?")
        search.assert_not_called()
        ids.assert_not_called()
        data = decode(self.state()["retrieval_context"])
        self.assertEqual(data["static_spaces"], [])
        self.assertNotIn("availability_options", data)
        self.assertEqual(data["dynamic_facts"]["prices"]["value"], {"hourly": 9})
        self.assertEqual(data["dynamic_facts"]["availability"], {"status": "missing"})

    def test_generic_turn_reuses_startup_static_evidence_but_reads_current_facts(self):
        self.dynamic.upsert_fact("demo-central", "prices", 7)
        with patch.object(self.static, "search", side_effect=AssertionError("Use baseline")):
            self.turn("Tell me about parking")
        data = decode(self.state()["retrieval_context"])
        self.assertEqual(data["static_spaces"], decode(self.agent.baseline_context)["static_spaces"])
        self.assertEqual(data["dynamic_facts"]["prices"]["value"], 7)

    def test_offered_ids_are_thread_scoped_and_not_publicly_writable(self):
        self.turn(thread="alice")
        alice_options = list(self.state("alice")["offered_space_ids"])
        self.dynamic.upsert_fact("demo-central", "availability", False)
        self.turn(thread="bob")
        self.assertEqual(self.state("bob")["offered_space_ids"], [])
        self.assertEqual(self.state("alice")["offered_space_ids"], alice_options)
        self.assertEqual(set(self.agent.graph.get_input_jsonschema()["properties"]), {"messages"})
        self.assertEqual(set(self.agent.graph.get_output_jsonschema()["properties"]), {"messages"})
        self.assertEqual(ParkingState().offered_space_ids, [])

    def test_initial_request_offers_options_without_model_response_or_hold(self):
        self.agent._reservation_extractor = RunnableLambda(lambda _: ReservationExtraction(intent="reservation"))
        result = self.turn("Reserve parking")
        self.assertIn("Current parking options:", result["messages"][-1].content)
        self.assertIn("not holds or bookings", result["messages"][-1].content)
        self.assertEqual(len(self.state()["offered_space_ids"]), 3)
        self.assertEqual(self.dynamic.list_facts("demo-central"), [])
        self.model.invoke.assert_not_called()
        self.turn("/cancel")
        self.assertEqual(self.state()["offered_space_ids"], [])

    def test_large_dynamic_evidence_keeps_valid_budget_and_tracks_only_emitted_options(self):
        config = ContextConfig(max_context_chars=2000)
        self.dynamic.upsert_fact("demo-central", "prices", "x" * 1800)
        context = retrieve_context([HumanMessage("Available slots?")], self.static, self.dynamic, config)
        self.assertLessEqual(len(context), 2000)
        data = decode(context)
        self.assertLessEqual(len(data["availability_options"]), 3)
        self.assertEqual(offered_space_ids(context), [item["space_id"] for item in data["availability_options"]])


class InventoryAndResetTests(ParkingFixture):
    def test_selected_reads_preserve_order_and_location_without_embedding(self):
        self.static.initialize()
        ids = self.static.list_space_ids("demo-central")[-3:][::-1]
        with patch.object(self.static._embeddings, "embed_query", side_effect=AssertionError("No embedding")):
            self.assertEqual([doc.id for doc in self.static.get_spaces(ids, location_id="demo-central")], ids)
        with self.assertRaises(RuntimeError):
            self.static.get_spaces(ids, location_id="other")
        with self.assertRaises(ValueError):
            self.static.get_spaces([ids[0], ids[0]], location_id="demo-central")
        self.assertEqual(self.static.get_spaces([], location_id="demo-central"), [])

    def test_new_collection_in_existing_directory_is_seeded_without_touching_other_collection(self):
        self.static.upsert_documents([document()])
        config = replace(self.storage, chroma_collection="another_collection")
        other = StaticParkingStore(config, embeddings=KeywordEmbeddings())
        other.initialize()
        self.assertEqual(len(other.list_space_ids("demo-central")), 20)
        self.assertEqual(len(self.static.list_space_ids("north")), 1)

    def test_sqlite_reset_recreates_corrupt_file_and_removes_sidecars_only(self):
        self.storage.sqlite_path.write_bytes(b"old or corrupt database")
        for suffix in ("-wal", "-shm", "-journal"):
            self.storage.sqlite_path.with_name(self.storage.sqlite_path.name + suffix).write_bytes(b"old sidecar")
        sentinel = self.root / "do-not-delete.txt"
        sentinel.write_text("Preserve unrelated files")
        self.dynamic.reset()
        self.assertEqual(self.dynamic.list_facts("demo-central"), [])
        self.assertTrue(sentinel.exists())
        self.assertFalse(self.storage.sqlite_path.with_name(self.storage.sqlite_path.name + "-wal").exists())

    def test_sqlite_cannot_be_configured_inside_persistent_chroma_directory(self):
        with self.assertRaisesRegex(ValueError, "outside the persistent Chroma"):
            StorageConfig(chroma_directory=self.root / "chroma", sqlite_path=self.root / "chroma/chroma.sqlite3")


class LifespanTests(ParkingFixture):
    def test_cancelled_startup_does_not_leave_a_ready_agent_or_running_preparation(self):
        async def run():
            started, release = Event(), Event()
            original = self.static.initialize

            def initialize():
                started.set()
                if not release.wait(5):
                    raise RuntimeError("Test did not release startup")
                original()

            async def start():
                async with app.lifespan(app.webapp):
                    self.fail("Cancelled startup must not accept interaction")

            with patch.object(app, "parking_agent", self.agent), patch.object(self.static, "initialize", initialize):
                task = asyncio.create_task(start())
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 2))
                    task.cancel()
                finally:
                    release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(self.agent.startup_ready)
                self.assertEqual(self.agent.baseline_context, "")

        asyncio.run(run())

    def test_real_starlette_lifespan_prepares_before_requests_and_resets_on_next_start(self):
        with patch.object(app, "parking_agent", self.agent):
            with TestClient(app.webapp) as client:
                self.assertTrue(self.agent.startup_ready)
                self.assertEqual(client.get("/").status_code, 404)  # No new public routes.
                self.dynamic.upsert_fact("demo-central", "availability", False)
            self.assertFalse(self.agent.startup_ready)
            with TestClient(app.webapp):
                self.assertTrue(self.agent.startup_ready)
                self.assertEqual(self.dynamic.list_facts("demo-central"), [])

    def test_lifespan_failure_prevents_serving(self):
        with (
            patch.object(app, "parking_agent", self.agent),
            patch.object(self.dynamic, "reset", side_effect=OSError("unwritable")),
            self.assertRaisesRegex(RuntimeError, "dynamic database reset"),
            TestClient(app.webapp),
        ):
            self.fail("Startup failure must prevent entering the client context")
        self.assertFalse(self.agent.startup_ready)

    def test_async_preparation_and_suggestions_stay_off_event_loop(self):
        async def run():
            loop_thread = current_thread()
            original_prepare = self.agent.prepare_startup
            original_get = self.static.get_spaces
            original_search = self.static.search

            def prepare():
                self.assertIsNot(current_thread(), loop_thread)
                original_prepare()

            def get_spaces(*args, **kwargs):
                self.assertIsNot(current_thread(), loop_thread)
                return original_get(*args, **kwargs)

            def search(*args, **kwargs):
                self.assertIsNot(current_thread(), loop_thread)
                return original_search(*args, **kwargs)

            with (
                patch.object(app, "parking_agent", self.agent),
                patch.object(self.agent, "prepare_startup", side_effect=prepare),
                patch.object(self.static, "get_spaces", side_effect=get_spaces),
                patch.object(self.static, "search", side_effect=search),
            ):
                async with app.lifespan(app.webapp):
                    results = await asyncio.gather(
                        *[
                            self.agent.graph.ainvoke(
                                {"messages": [HumanMessage("Available spaces?")]}, {"configurable": {"thread_id": name}}
                            )
                            for name in ("one", "two")
                        ]
                    )
                    self.assertEqual(len(results), 2)
                    self.assertEqual(len(self.state("one")["offered_space_ids"]), 3)
                    self.assertEqual(len(self.state("two")["offered_space_ids"]), 3)
            self.assertFalse(self.agent.startup_ready)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
