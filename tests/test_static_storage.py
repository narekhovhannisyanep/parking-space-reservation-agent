"""Real persistent Chroma checks using small offline embeddings, never provider calls."""

import json
import os
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings
from openai import APIError, OpenAIError

from storage import ParkingSpace, SpaceFilters, StaticParkingStore, StorageConfig
from storage.static import _LazyAzureEmbeddings


class KeywordEmbeddings(Embeddings):
    """Tiny deterministic vectors to test retrieval, not measure semantic accuracy."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float("entrance" in text.lower()), float("lift" in text.lower()), 0.1]


def document(identifier="entrance", text="Space beside the entrance.", location="north", **changes):
    record = {
        "space_id": identifier,
        "location_id": location,
        "label": identifier,
        "description": text,
        "source": "reviewed-space-inventory",
        "hourly_price_cents": 500,
    }
    return ParkingSpace.from_record({**record, **changes}).to_document()


class StaticStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = StorageConfig(chroma_directory=Path(self.directory.name) / "nested/chroma")
        self.store = StaticParkingStore(self.config, embeddings=KeywordEmbeddings())

    def test_construction_and_empty_mutations_do_not_create_resources(self):
        self.assertEqual(self.store.upsert_documents([]), [])
        self.store.delete_documents([])
        self.assertFalse(self.config.chroma_directory.exists())

    def test_empty_database_and_missing_location_do_not_initialize_embeddings(self):
        with patch.dict(os.environ, {}, clear=True), patch("storage.static.AzureOpenAIEmbeddings") as constructor:
            store = StaticParkingStore(self.config)
            self.assertEqual(store.search("entrance", location_id="north"), [])
            constructor.assert_not_called()
        self.assertTrue(self.config.chroma_directory.exists())

    def test_semantic_search_filters_locations_and_clamps_k(self):
        self.store.upsert_documents(
            [
                document(),
                document("lift", "Space beside the lift."),
                document("south", "Space beside the entrance.", location="south"),
            ]
        )
        result = self.store.search("entrance", location_id="north", k=10)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, "north:space:entrance")
        self.assertTrue(all(doc.metadata["location_id"] == "north" for doc in result))
        self.assertEqual(self.store.search("entrance", location_id="unknown"), [])

    def test_upsert_is_idempotent_and_deletion_is_explicit(self):
        self.store.upsert_documents([document()])
        self.assertEqual(
            self.store.upsert_documents([document(text="Space beside the east entrance.")]), ["north:space:entrance"]
        )
        result = self.store.search("entrance", location_id="north")
        self.assertEqual(len(result), 1)
        self.assertIn("Space beside the east entrance.", result[0].page_content)
        self.store.delete_documents([])
        self.assertEqual(len(self.store.search("entrance", location_id="north")), 1)
        self.store.delete_documents(["north:space:entrance"])
        self.assertEqual(self.store.search("entrance", location_id="north"), [])

    def test_data_survives_a_new_instance_and_process(self):
        self.store.upsert_documents([document()])
        reopened = StaticParkingStore(self.config, embeddings=KeywordEmbeddings())
        self.assertEqual(reopened.search("entrance", location_id="north")[0].page_content, document().page_content)
        code = (
            "import chromadb, json, sys; from chromadb.config import Settings; "
            "c = chromadb.PersistentClient(path=sys.argv[1], settings=Settings(anonymized_telemetry=False)); "
            "print(json.dumps(c.get_collection(sys.argv[2], embedding_function=None).get()['documents']))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(self.config.chroma_directory), self.config.chroma_collection],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(result.stdout), [document().page_content])

    def test_schema_migration_cannot_overwrite_existing_same_id(self):
        legacy = document()
        identifier = legacy.id
        assert identifier is not None
        legacy.metadata["schema_version"] = 2
        del legacy.metadata["hourly_price_cents"]
        self.store._get_store().add_documents([legacy])
        with (
            patch.object(self.store._embeddings, "embed_documents", side_effect=AssertionError("must not reembed")),
            self.assertRaisesRegex(ValueError, "fresh Chroma collection"),
        ):
            self.store.upsert_documents([document()])
        self.assertEqual(
            self.store._get_store().get(ids=[identifier], include=["metadatas"])["metadatas"], [legacy.metadata]
        )

    def test_distinct_collections_are_isolated(self):
        self.store.upsert_documents([document()])
        config = StorageConfig(chroma_directory=self.config.chroma_directory, chroma_collection="other_static")
        other = StaticParkingStore(config, embeddings=KeywordEmbeddings())
        self.assertEqual(other.search("entrance", location_id="north"), [])

    def test_invalid_documents_reject_the_whole_batch_before_creating_resources(self):
        invalid = []
        for name, value in (
            ("location_id", " "),
            ("vehicle_type", "spaceship"),
            ("accessible", "true"),
            ("width_m", -1),
            ("length_m", float("nan")),
            ("max_height_m", True),
            ("schema_version", True),
            ("record_type", "booking_process"),
        ):
            bad = document()
            bad.metadata[name] = value
            invalid.append(bad)
        wrong_id = document()
        wrong_id.id = "north:space:different"
        invalid.append(wrong_id)
        blank_text = document()
        blank_text.page_content = " "
        invalid.append(blank_text)
        missing_id = document()
        missing_id.id = None
        invalid.append(missing_id)
        extra_metadata = document()
        extra_metadata.metadata["car_number"] = "private"
        invalid.append(extra_metadata)
        missing_source = document()
        del missing_source.metadata["source"]
        invalid.append(missing_source)
        for doc in invalid:
            with self.subTest(doc=doc), self.assertRaises(ValueError):
                self.store.upsert_documents([document("valid"), doc])
        with self.assertRaises(ValueError):
            self.store.upsert_documents([document(), document()])
        self.assertFalse(self.config.chroma_directory.exists())

    def test_exact_space_filter_applies_before_ranking_and_preserves_location(self):
        self.store.upsert_documents(
            [
                document(),
                document("lift", "Space beside the lift."),
                document("lift", location="south"),
            ]
        )
        results = self.store.search("entrance", location_id="north", filters=SpaceFilters(space_id="lift"))
        self.assertEqual([doc.id for doc in results], ["north:space:lift"])
        self.assertIn("Space beside the lift.", results[0].page_content)
        self.assertEqual(len(self.store.search("entrance", location_id="north", filters=SpaceFilters())), 2)

    def test_no_matching_constraints_do_not_embed(self):
        self.store.upsert_documents([document()])
        with patch.object(self.store._embeddings, "embed_query", side_effect=AssertionError("must not embed")):
            self.assertEqual(
                self.store.search("lift", location_id="north", filters=SpaceFilters(space_id="missing")), []
            )

    def test_eligible_document_filter_applies_before_ranking_and_preserves_scope(self):
        self.store.upsert_documents(
            [document(), document("lift", "Space beside the lift."), document("lift", location="south")]
        )
        legacy = document("legacy", "Space beside the entrance.")
        legacy.metadata["schema_version"] = 1
        self.store._get_store().add_documents([legacy])
        result = self.store.search(
            "entrance", location_id="north", k=3, eligible_ids=["north:space:lift", "north:space:legacy"]
        )
        self.assertEqual([doc.id for doc in result], ["north:space:lift"])
        self.assertEqual(
            self.store.search(
                "entrance",
                location_id="north",
                filters=SpaceFilters(space_id="entrance"),
                eligible_ids=["north:space:lift"],
            ),
            [],
        )

    def test_empty_or_nonmatching_eligibility_does_not_embed(self):
        self.assertEqual(self.store.search("lift", location_id="north", eligible_ids=[]), [])
        self.assertFalse(self.config.chroma_directory.exists())
        self.store.upsert_documents([document()])
        with patch.object(self.store._embeddings, "embed_query", side_effect=AssertionError("must not embed")):
            self.assertEqual(self.store.search("lift", location_id="north", eligible_ids=["north:space:missing"]), [])

    def test_invalid_eligibility_cannot_broaden_scope_or_initialize_storage(self):
        for ids in (["south:space:lift"], ["north:overview"], ["north:space:../bad"], ["north:space:lift"] * 2):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.store.search("lift", location_id="north", eligible_ids=ids)
        self.assertFalse(self.config.chroma_directory.exists())

    def test_legacy_guide_records_are_never_retrieved_even_in_same_collection(self):
        # Simulate old persisted content by bypassing the new trusted-ingestion validator.
        self.store._get_store().add_documents(
            [
                Document(
                    id="north:overview",
                    page_content="The entrance is here.",
                    metadata={"location_id": "north", "source": "old-guide", "topic": "general"},
                )
            ]
        )
        self.assertEqual(self.store.search("entrance", location_id="north"), [])
        self.store.upsert_documents([document()])
        self.assertEqual(
            [doc.id for doc in self.store.search("entrance", location_id="north")], ["north:space:entrance"]
        )

    def test_arbitrary_filters_cannot_override_location_or_schema(self):
        invalid_filters: Any = {"location_id": "south"}
        with self.assertRaises(TypeError):
            self.store.search("entrance", location_id="north", filters=invalid_filters)
        self.assertFalse(self.config.chroma_directory.exists())

    def test_obsolete_inventory_schema_is_not_retrieved_or_accepted(self):
        legacy = document("legacy")
        legacy.metadata.update(schema_version=1, vehicle_type="car", length_m=5.0)
        self.store._get_store().add_documents([legacy])
        with patch.object(self.store._embeddings, "embed_query", side_effect=AssertionError("must not embed")):
            self.assertEqual(self.store.search("entrance", location_id="north"), [])
        with self.assertRaises(ValueError):
            self.store.upsert_documents([legacy])
        self.store.upsert_documents([document()])
        self.assertEqual(
            [doc.id for doc in self.store.search("entrance", location_id="north")], ["north:space:entrance"]
        )

    def test_upsert_preserves_minimal_metadata_and_caller_document(self):
        original = document()
        expected = document()
        self.store.upsert_documents([original])
        self.assertEqual(original, expected)
        result = self.store.search("semi or bicycle", location_id="north")[0]
        self.assertEqual(result, expected)
        ParkingSpace.validate_document(result)

    def test_embedding_failure_does_not_remove_existing_space_or_metadata(self):
        original = document()
        self.store.upsert_documents([original])
        with (
            patch.object(self.store._embeddings, "embed_documents", side_effect=RuntimeError("offline")),
            self.assertRaisesRegex(RuntimeError, "offline"),
        ):
            self.store.upsert_documents([document(text="Updated directions.")])
        self.assertEqual(self.store.search("entrance", location_id="north"), [original])

    def test_invalid_queries_and_deletes_are_rejected(self):
        invalid: list[tuple[str, str, Any]] = [
            ("", "north", 1),
            ("entrance", " ", 1),
            ("entrance", "north", 0),
            ("entrance", "north", True),
            ("entrance", "north", 1.5),
        ]
        for query, location, k in invalid:
            with self.subTest(query=query, location=location, k=k), self.assertRaises(ValueError):
                self.store.search(query, location_id=location, k=k)
        for ids in ([" "], ["duplicate", "duplicate"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.store.delete_documents(ids)
        self.assertFalse(self.config.chroma_directory.exists())

    def test_concurrent_initialization_reuses_one_store(self):
        with patch("storage.static.Chroma") as constructor, ThreadPoolExecutor(max_workers=8) as executor:
            resources = list(executor.map(lambda _: self.store._get_store(), range(16)))
        constructor.assert_called_once()
        self.assertTrue(all(resource is resources[0] for resource in resources))

    def test_store_initialization_failure_can_be_retried(self):
        resource = Mock()
        with patch("storage.static.Chroma", side_effect=[OSError("unavailable disk"), resource]) as constructor:
            with self.assertRaises(OSError):
                self.store._get_store()
            self.assertIsNone(self.store._store)
            self.assertIs(self.store._get_store(), resource)
        self.assertEqual(constructor.call_count, 2)


class LazyEmbeddingTests(unittest.TestCase):
    def test_real_client_accepts_azure_environment_without_network_calls(self):
        env = {
            "AZURE_OPENAI_API_KEY": "offline-placeholder",
            "AZURE_OPENAI_ENDPOINT": "https://offline.example.invalid/",
            "OPENAI_API_VERSION": "2024-10-21",
        }
        with patch.dict(os.environ, env, clear=True):
            client = _LazyAzureEmbeddings(StorageConfig())._get_client()
        assert isinstance(client, AzureOpenAIEmbeddings)
        self.assertEqual(client.deployment, StorageConfig().embedding_deployment)
        self.assertEqual(client.model, StorageConfig().embedding_model)
        self.assertEqual(client.openai_api_version, env["OPENAI_API_VERSION"])
        self.assertFalse(client.check_embedding_ctx_length)

    def test_missing_environment_is_sanitized(self):
        with patch.dict(os.environ, {}, clear=True):
            embeddings = _LazyAzureEmbeddings(StorageConfig())
            with self.assertRaisesRegex(RuntimeError, "Unable to initialize embeddings"):
                embeddings.embed_query("query")

    def test_provider_initializes_once_with_configured_deployment_and_no_tokenization(self):
        config = StorageConfig(embedding_deployment="custom", embedding_model="custom-model")
        with patch("storage.static.AzureOpenAIEmbeddings") as constructor:
            embeddings = _LazyAzureEmbeddings(config)
            constructor.assert_not_called()
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(embeddings.embed_query, ["query"] * 16))
            embeddings.embed_documents(["static document"])
        constructor.assert_called_once_with(
            azure_deployment="custom",
            model="custom-model",
            check_embedding_ctx_length=False,
            request_timeout=60,
            max_retries=2,
        )

    def test_initialization_failure_is_sanitized_and_retryable(self):
        client = Mock()
        client.embed_query.return_value = [1.0]
        with patch("storage.static.AzureOpenAIEmbeddings", side_effect=[OpenAIError("secret"), client]):
            embeddings = _LazyAzureEmbeddings(StorageConfig())
            with self.assertRaisesRegex(RuntimeError, "Unable to initialize embeddings") as error:
                embeddings.embed_query("query")
            self.assertNotIn("secret", str(error.exception))
            self.assertTrue(error.exception.__suppress_context__)
            self.assertEqual(embeddings.embed_query("query"), [1.0])

    def test_api_errors_are_sanitized_but_programming_errors_propagate(self):
        client = Mock()
        client.embed_query.side_effect = APIError("secret", request=Mock(), body=None)
        client.embed_documents.side_effect = APIError("secret", request=Mock(), body=None)
        with patch("storage.static.AzureOpenAIEmbeddings", return_value=client):
            embeddings = _LazyAzureEmbeddings(StorageConfig())
            for operation in (lambda: embeddings.embed_query("query"), lambda: embeddings.embed_documents(["doc"])):
                with self.assertRaises(RuntimeError) as error:
                    operation()
                self.assertNotIn("secret", str(error.exception))
            client.embed_query.side_effect = TypeError("bug")
            with self.assertRaisesRegex(TypeError, "bug"):
                embeddings.embed_query("query")
