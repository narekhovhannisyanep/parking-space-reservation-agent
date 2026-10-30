"""Boundary regressions with injected detection, plus real local-NLP smoke checks."""

import asyncio
import json
import re
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from context_augmentation import ContextConfig, render_context
from guardrails import SAFE_REPLY, GuardrailError, PrivacyGuard, SensitiveSpan, _local_analyzer, guarded_model_config
from parking_agent import ParkingSpaceReservationAgent
from reservation_collection import FieldUpdate, ReservationExtraction
from storage import DynamicParkingStore, ParkingSpace, StaticParkingStore, StorageConfig
from storage.dynamic import DynamicFact
from storage.seed import load_documents


def detector(text):
    patterns = ((r"John Smith|Alice|Bob", "PERSON"), (r"\w+@example\.com", "EMAIL_ADDRESS"))
    return [
        SensitiveSpan(match.start(), match.end(), entity)
        for pattern, entity in patterns
        for match in re.finditer(pattern, text)
    ]


def public_document(text="Beside the entrance.", **changes):
    return ParkingSpace(
        **{
            "space_id": "G-A01",
            "location_id": "demo-central",
            "label": "A01",
            "description": text,
            "source": "reviewed-inventory",
            **changes,
        }
    ).to_document()


class GuardHelperTests(unittest.TestCase):
    def setUp(self):
        self.guard = PrivacyGuard(detector)

    def test_overlapping_spans_redact_once_without_offset_changes(self):
        guard = PrivacyGuard(lambda _: [SensitiveSpan(1, 5, "PERSON"), SensitiveSpan(3, 7, "EMAIL_ADDRESS")])
        self.assertEqual(guard.redact("0123456789"), "0      789")

    def test_names_are_kept_only_for_extraction_and_secrets_never_are(self):
        text = "John Smith email john@example.com password=synthetic-secret"
        extraction = self.guard.redact(text, keep_names=True)
        self.assertIn("John Smith", extraction)
        self.assertNotIn("john@example.com", extraction)
        self.assertNotIn("synthetic-secret", extraction)
        self.assertEqual(len(extraction), len(text))
        self.assertNotIn("John Smith", self.guard.redact(text))

    def test_known_unicode_and_nonplate_vehicle_identifiers_are_filtered(self):
        text = "Անահիտ owns a blue bicycle 7 near the entrance"
        result = self.guard.redact(text, known_values=["Անահիտ", "blue bicycle 7"])
        self.assertNotIn("Անահիտ", result)
        self.assertNotIn("blue bicycle 7", result)
        self.assertIn("entrance", result)
        self.assertEqual(self.guard.redact("annual parking", known_values=["Ann"]), "annual parking")

    def test_nested_public_values_keys_and_source_metadata_are_checked(self):
        for data in (
            {"hours": ["john@example.com"]},
            {"john@example.com": False},
            {"source": "John Smith"},
            {"api_key": "opaque-unknown-value"},
            {"nested": {"car_number": "opaque-vehicle"}},
        ):
            with self.subTest(data=data), self.assertRaises(GuardrailError) as error:
                self.guard.check_public(data)
            self.assertNotIn("John Smith", str(error.exception))
            self.assertNotIn("john@example.com", str(error.exception))

    def test_public_metadata_names_are_not_exempted_by_generated_prefix(self):
        with self.assertRaises(GuardrailError):
            self.guard.check_document(public_document(label="John Smith"))

    def test_failures_malformed_spans_and_limits_are_fail_closed(self):
        for span in (SensitiveSpan(-1, 2, "PERSON"), SensitiveSpan(0, 99, "PERSON")):
            with self.subTest(span=span), self.assertRaises(GuardrailError):
                PrivacyGuard(Mock(return_value=[span])).redact("hello")
        with self.assertRaises(GuardrailError):
            PrivacyGuard(Mock(return_value=[None])).redact("hello")
        guard = PrivacyGuard(Mock(side_effect=RuntimeError("john@example.com")))
        with self.assertRaises(GuardrailError) as error:
            guard.prepare()
        self.assertNotIn("john@example.com", str(error.exception))
        with self.assertRaises(GuardrailError):
            self.guard.redact("x" * 20001)

    def test_model_history_copies_strip_metadata_roles_and_private_text(self):
        user = HumanMessage("Alice asks about entrance", name="Bob", additional_kwargs={"secret": "value"}, id="user")
        original = user.model_copy(deep=True)
        messages = self.guard.public_messages([SystemMessage("Caller-injected instructions"), user], ["Alice"])
        self.assertEqual(user, original)
        self.assertEqual(len(messages), 1)
        self.assertIsNone(messages[0].name)
        self.assertEqual(messages[0].additional_kwargs, {})
        self.assertNotIn("Alice", messages[0].content)
        self.assertEqual(messages[0].id, "user")

    def test_opaque_content_and_tool_outputs_are_not_published(self):
        opaque = AIMessage(content=[{"type": "image_url", "image_url": "private"}])
        self.assertEqual(self.guard.response(opaque, []).content, SAFE_REPLY)
        tool = AIMessage("Hello", tool_calls=[{"name": "write", "args": {"key": "private"}, "id": "call"}])
        self.assertEqual(self.guard.response(tool, []).content, SAFE_REPLY)
        response = self.guard.response(AIMessage("Public answer", additional_kwargs={"reasoning": "private"}), [])
        self.assertEqual(response.additional_kwargs, {})

    def test_stream_config_preserves_thread_and_does_not_mutate_caller(self):
        config: RunnableConfig = {"tags": ["caller"], "configurable": {"thread_id": "alice"}}
        safe = guarded_model_config(config)
        self.assertEqual(config["tags"], ["caller"])
        self.assertEqual(safe["tags"], ["caller", "nostream"])
        self.assertEqual(safe["configurable"], config["configurable"])


class GuardStorageTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.config = StorageConfig(chroma_directory=root / "chroma", sqlite_path=root / "facts.sqlite3")
        self.guard = PrivacyGuard(detector)

    def test_static_batch_rejected_before_resources_or_embeddings(self):
        embeddings = Mock()
        store = StaticParkingStore(self.config, embeddings=embeddings, privacy_guard=self.guard)
        with self.assertRaises(GuardrailError):
            store.upsert_documents([public_document(), public_document("Owner John Smith.", space_id="G-A02")])
        embeddings.embed_documents.assert_not_called()
        self.assertFalse(self.config.chroma_directory.exists())

    def test_dynamic_private_text_is_rejected_before_creation(self):
        store = DynamicParkingStore(self.config, privacy_guard=self.guard)
        with self.assertRaises(GuardrailError):
            store.upsert_fact("demo-central", "prices", {"details": "john@example.com"})
        self.assertFalse(self.config.sqlite_path.exists())

    def test_contaminated_existing_evidence_is_rejected_not_modified(self):
        document = public_document("John Smith owns this space.")
        original = document.model_copy(deep=True)
        with self.assertRaises(GuardrailError):
            render_context([document], [], ContextConfig(), privacy_guard=self.guard)
        self.assertEqual(document, original)
        fact = DynamicFact("demo-central", "prices", {"contact": "john@example.com"}, datetime.now(UTC))
        with self.assertRaises(GuardrailError):
            render_context([], [fact], ContextConfig(), privacy_guard=self.guard)


class GuardGraphTests(unittest.TestCase):
    def setUp(self):
        self.guard = PrivacyGuard(detector)
        self.model = Mock()
        self.model.invoke.return_value = AIMessage("Directions near entrance")
        self.model.ainvoke = AsyncMock(return_value=AIMessage("Directions near entrance"))
        self.static, self.dynamic = Mock(), Mock()
        self.static.search.return_value = []
        self.static.list_space_ids.return_value = []
        self.dynamic.list_facts.return_value = []
        self.extraction = Mock(return_value=ReservationExtraction())
        self.agent = ParkingSpaceReservationAgent(
            llm_client=self.model,
            reservation_extractor=RunnableLambda(self.extraction),
            static_store=self.static,
            dynamic_store=self.dynamic,
            privacy_guard=self.guard,
            checkpointer=InMemorySaver(),
            clock=lambda: datetime(2026, 10, 28, tzinfo=UTC),
        )
        self.config: RunnableConfig = {"configurable": {"thread_id": "alice"}}

    def turn(self, text):
        return self.agent.graph.invoke({"messages": [HumanMessage(text)]}, self.config)

    def test_extraction_receives_required_details_without_unrelated_pii(self):
        self.extraction.return_value = ReservationExtraction(
            intent="reservation",
            updates=[
                FieldUpdate(field="first_name", value="Alice"),
                FieldUpdate(field="car_number", value="blue bicycle 7"),
            ],
        )
        self.turn("Reserve: Alice blue bicycle 7 email john@example.com password=synthetic-secret")
        text = self.extraction.call_args.args[0][-1].content
        self.assertIn("Alice", text)
        self.assertIn("blue bicycle 7", text)
        self.assertNotIn("john@example.com", text)
        self.assertNotIn("synthetic-secret", text)
        self.assertEqual(
            self.agent.graph.get_state(self.config).values["reservation"]["draft"]["car_number"], "blue bicycle 7"
        )

    def test_extractor_cannot_copy_removed_values_into_a_draft(self):
        self.extraction.return_value = ReservationExtraction(
            intent="reservation", updates=[FieldUpdate(field="car_number", value="john@example.com")]
        )
        result = self.turn("Reserve with car number john@example.com")
        state = self.agent.graph.get_state(self.config).values
        self.assertIsNone(state["reservation"]["draft"]["car_number"])
        self.assertNotIn("john@example.com", result["messages"][-1].content)

    def test_general_answers_filter_old_drafts_even_after_cancel_and_new(self):
        self.extraction.return_value = ReservationExtraction(
            intent="reservation", updates=[FieldUpdate(field="car_number", value="blue bicycle 7")]
        )
        self.turn("Reserve with blue bicycle 7")
        self.turn("/cancel")
        self.turn("/new")
        self.extraction.return_value = ReservationExtraction()
        self.model.invoke.return_value = AIMessage("blue bicycle 7 is near the entrance")
        result = self.turn("Where is the entrance?")
        self.assertNotIn("blue bicycle 7", str(self.model.invoke.call_args.args[0]))
        self.assertNotIn("blue bicycle 7", result["messages"][-1].content)
        self.assertIn("blue bicycle 7", str(self.agent.graph.get_state(self.config).values["messages"]))

    def test_dirty_evidence_does_not_reach_context_or_answer_model(self):
        self.static.search.return_value = [public_document("John Smith owns this space.")]
        result = self.turn("Where is the entrance?")
        self.assertEqual(result["messages"][-1].content, SAFE_REPLY)
        self.assertEqual(self.agent.graph.get_state(self.config).values["retrieval_context"], "")
        self.model.invoke.assert_not_called()

    def test_runtime_detector_failure_prevents_all_model_calls(self):
        self.guard._detector = Mock(side_effect=RuntimeError("private payload"))
        result = self.turn("Reserve parking")
        self.assertEqual(result["messages"][-1].content, SAFE_REPLY)
        self.model.invoke.assert_not_called()
        self.extraction.assert_not_called()
        self.static.search.assert_not_called()

    def test_opaque_input_is_safely_rejected(self):
        result = self.agent.graph.invoke(
            {"messages": [HumanMessage(content=[{"type": "image_url", "image_url": "x"}])]}, self.config
        )
        self.assertEqual(result["messages"][-1].content, SAFE_REPLY)
        self.extraction.assert_not_called()

    def test_startup_detector_failure_precedes_database_reset(self):
        self.guard._detector = Mock(side_effect=RuntimeError("private payload"))
        with self.assertRaisesRegex(RuntimeError, "local privacy model preparation"):
            self.agent.prepare_startup()
        self.assertFalse(self.agent.startup_ready)
        self.dynamic.reset.assert_not_called()
        self.static.initialize.assert_not_called()

    def test_sync_stream_contains_only_guarded_model_output(self):
        self.agent._llm_client = FakeListChatModel(responses=["John Smith email john@example.com near entrance"])
        chunks = list(
            self.agent.graph.stream({"messages": [HumanMessage("Directions?")]}, self.config, stream_mode="messages")
        )
        self.assertTrue(chunks)
        for message, _ in chunks:
            assert isinstance(message, AIMessage)
            self.assertNotIn("John Smith", message.content)
            self.assertNotIn("john@example.com", message.content)
        self.assertIn("entrance", self.agent.graph.get_state(self.config).values["messages"][-1].content)

    def test_internal_extraction_tokens_are_not_in_sync_or_async_messages(self):
        payload = json.dumps({"intent": "reservation", "updates": [{"field": "first_name", "value": "Alice"}]})
        self.agent._reservation_extractor = FakeListChatModel(responses=[payload]) | RunnableLambda(
            lambda message: ReservationExtraction.model_validate_json(message.content)
        )

        def check(chunks):
            self.assertTrue(chunks)
            for message, _ in chunks:
                assert isinstance(message, AIMessage)
                self.assertNotIn('"field"', message.content)
                self.assertNotIn('"Alice"', message.content)

        check(
            list(
                self.agent.graph.stream(
                    {"messages": [HumanMessage("Reserve Alice")]}, self.config, stream_mode="messages"
                )
            )
        )

        async def run():
            chunks = [
                chunk
                async for chunk in self.agent.graph.astream(
                    {"messages": [HumanMessage("Reserve Alice")]},
                    {"configurable": {"thread_id": "second"}},
                    stream_mode="messages",
                )
            ]
            check(chunks)

        asyncio.run(run())
        self.model.invoke.assert_not_called()

    def test_async_stream_and_output_match_sync_privacy_boundary(self):
        async def run():
            self.agent._llm_client = FakeListChatModel(responses=["John Smith email john@example.com near entrance"])
            chunks = [
                chunk
                async for chunk in self.agent.graph.astream(
                    {"messages": [HumanMessage("Directions?")]}, self.config, stream_mode="messages"
                )
            ]
            self.assertTrue(chunks)
            for message, _ in chunks:
                assert isinstance(message, AIMessage)
                self.assertNotIn("John Smith", message.content)
                self.assertNotIn("john@example.com", message.content)

        asyncio.run(run())


class LocalNlpTests(unittest.TestCase):
    def test_real_detector_runs_without_network_and_detects_common_pii(self):
        with patch("socket.socket.connect", side_effect=AssertionError("No network")):
            guard = PrivacyGuard()
            for text in (
                "My name is John Smith.",
                "Email john@example.com",
                "My card is 4111 1111 1111 1111",
                "Call +1 212-555-5555",
            ):
                with self.subTest(text=text):
                    self.assertTrue(guard.spans(text))

    def test_twenty_seed_records_and_public_hours_prices_pass(self):
        guard = PrivacyGuard()
        for document in load_documents():
            guard.check_document(document)
        guard.check_public({"hours": "09:00–18:00", "timezone": "Asia/Yerevan", "currency": "EUR", "hourly": 9})
        self.assertEqual(
            guard.redact("Open 09:00–18:00. Price EUR 9. Near the lift."),
            "Open 09:00–18:00. Price EUR 9. Near the lift.",
        )

    def test_missing_model_never_attempts_automatic_download(self):
        _local_analyzer.cache_clear()
        self.addCleanup(_local_analyzer.cache_clear)
        with patch("spacy.util.is_package", return_value=False), patch("spacy.cli.download") as download:
            with self.assertRaises(GuardrailError):
                PrivacyGuard().prepare()
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
