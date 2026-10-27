"""Seed validation and real Chroma ingestion with offline embeddings only."""

import io
import json
import os
import subprocess
import sys
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.documents import Document
from test_static_storage import KeywordEmbeddings

from storage import StaticParkingStore, StorageConfig
from storage.seed import DEFAULT_SEED_FILE, load_documents, main

EXPECTED_SEED_COUNT = 30


class StaticSeedTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_fixture(self, records):
        path = self.root / "seed.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        return path

    def test_default_fixture_has_thirty_unique_short_static_records(self):
        documents = load_documents()
        self.assertEqual(len(documents), EXPECTED_SEED_COUNT)
        self.assertEqual(len({doc.id for doc in documents}), EXPECTED_SEED_COUNT)
        self.assertEqual(len({doc.page_content for doc in documents}), EXPECTED_SEED_COUNT)
        self.assertEqual({doc.metadata["location_id"] for doc in documents}, {"demo-central"})
        self.assertEqual(
            Counter(doc.metadata["topic"] for doc in documents),
            {"general": 10, "location": 9, "booking_process": 11},
        )
        for doc in documents:
            assert doc.id is not None
            self.assertTrue(doc.id.startswith("demo-central:"))
            self.assertNotIn("synthetic", doc.page_content.lower())
            self.assertNotIn("fictional", doc.page_content.lower())
            self.assertTrue(doc.page_content.strip())
            self.assertLess(len(doc.page_content), 700)
            self.assertEqual(doc.metadata["source"], "demo-parking-guide-v1")
            self.assertEqual(set(doc.metadata), {"location_id", "source", "topic"})
        ids = {doc.id for doc in documents}
        for scenario in (
            "incomplete-request",
            "pending-approval",
            "refused-request",
            "approved-request",
            "recording-outcome",
            "speed-limit",
            "bay-use",
            "belongings",
            "toilets",
            "lift-location",
            "motorcycle-area",
            "period-clarification",
            "details-correction",
            "reservation-privacy",
            "payment-vs-approval",
        ):
            self.assertIn(f"demo-central:{scenario}", ids)

    def test_additional_records_cover_distinct_facts_and_workflow_boundaries(self):
        content = {doc.id: doc.page_content for doc in load_documents()}
        expected_facts = {
            "speed-limit": "10 km/h",
            "bay-use": "marked bay boundaries",
            "belongings": "does not provide luggage lockers",
            "toilets": "including an accessible toilet",
            "lift-location": "north-east corner",
            "motorcycle-area": "along the south wall",
            "period-clarification": "no default timezone or maximum stay",
            "details-correction": "not administrator approval",
            "reservation-privacy": "private reservation information",
            "payment-vs-approval": "not a human administrator's approval",
        }
        for identifier, fact in expected_facts.items():
            with self.subTest(identifier=identifier):
                self.assertIn(fact, content[f"demo-central:{identifier}"])

    def test_loading_returns_independent_documents_and_does_not_touch_storage(self):
        with patch("storage.static.Chroma") as chroma, patch("storage.static.AzureOpenAIEmbeddings") as embeddings:
            first = load_documents()
            first[0].metadata["location_id"] = "changed"
            self.assertEqual(load_documents()[0].metadata["location_id"], "demo-central")
        chroma.assert_not_called()
        embeddings.assert_not_called()

    def test_invalid_shapes_metadata_topics_and_duplicates_are_rejected(self):
        valid = json.loads(DEFAULT_SEED_FILE.read_text(encoding="utf-8"))[0]
        bad_records = [None, {}, [], [None], [{"id": "one"}]]
        for field, value in (("id", 1), ("id", " "), ("page_content", None), ("page_content", " "), ("metadata", [])):
            bad_records.append([{**valid, field: value}])
        for metadata in (
            {},
            {**valid["metadata"], "topic": "availability"},
            {**valid["metadata"], "location_id": " "},
            {**valid["metadata"], "source": 123},
            {**valid["metadata"], "car_number": "private"},
        ):
            bad_records.append([valid, {**valid, "id": "second", "metadata": metadata}])
        bad_records.extend([[valid, valid], [{**valid, "extra": "not allowed"}]])
        with patch("storage.static.Chroma") as chroma, patch("storage.static.AzureOpenAIEmbeddings") as embeddings:
            for records in bad_records:
                with self.subTest(records=records), self.assertRaises(ValueError):
                    load_documents(self.write_fixture(records))
        chroma.assert_not_called()
        embeddings.assert_not_called()

    def test_missing_and_malformed_files_fail(self):
        with self.assertRaises(FileNotFoundError):
            load_documents(self.root / "missing.json")
        path = self.root / "bad.json"
        path.write_text("not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_documents(path)

    def test_custom_fixture_can_grow_beyond_bundled_record_count(self):
        records = json.loads(DEFAULT_SEED_FILE.read_text(encoding="utf-8"))
        records.append({**records[0], "id": "demo-central:additional-example"})
        self.assertEqual(len(load_documents(self.write_fixture(records))), EXPECTED_SEED_COUNT + 1)

    def test_dry_run_needs_no_environment_or_resources(self):
        output = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("storage.seed.StaticParkingStore", wraps=StaticParkingStore) as constructor,
            patch("storage.seed.StorageConfig.from_environment") as config,
            redirect_stdout(output),
        ):
            self.assertEqual(main(["--dry-run"]), 0)
        constructor.assert_not_called()
        config.assert_not_called()
        self.assertIn(f"Validated {EXPECTED_SEED_COUNT} static records", output.getvalue())
        self.assertIn("No data written", output.getvalue())

    def test_cli_seed_is_repeatable_preserves_other_records_and_persists(self):
        config = StorageConfig(chroma_directory=self.root / "chroma", sqlite_path=self.root / "dynamic.sqlite3")
        store = StaticParkingStore(config, embeddings=KeywordEmbeddings())
        unrelated = Document(
            id="other:entrance",
            page_content="Other location entrance.",
            metadata={"location_id": "other", "source": "test", "topic": "location"},
        )
        store.upsert_documents([unrelated])
        output = io.StringIO()
        with (
            patch("storage.seed.StorageConfig.from_environment", return_value=config),
            patch("storage.seed.StaticParkingStore", wraps=StaticParkingStore) as constructor,
            redirect_stdout(output),
        ):
            constructor.return_value = store
            self.assertEqual(main([]), 0)
            self.assertEqual(main([]), 0)
        self.assertIn(f"Upserted {EXPECTED_SEED_COUNT} static records", output.getvalue())
        reopened = StaticParkingStore(config, embeddings=KeywordEmbeddings())
        results = reopened.search("entrance", location_id="demo-central", k=100)
        self.assertEqual({doc.id for doc in results}, {doc.id for doc in load_documents()})
        self.assertEqual(len(results), EXPECTED_SEED_COUNT)
        self.assertTrue(all(doc.metadata["location_id"] == "demo-central" for doc in results))
        self.assertEqual(reopened.search("entrance", location_id="other")[0].id, unrelated.id)
        self.assertEqual(reopened.search("entrance", location_id="unknown"), [])
        self.assertFalse(config.sqlite_path.exists())
        changed = load_documents()[0]
        changed.page_content = "Updated parking overview."
        reopened.upsert_documents([changed])
        reopened.upsert_documents(load_documents())
        results = reopened.search("entrance", location_id="demo-central", k=100)
        self.assertEqual(len(results), EXPECTED_SEED_COUNT)
        self.assertEqual(
            next(doc.page_content for doc in results if doc.id == changed.id), load_documents()[0].page_content
        )

    def test_cli_invalid_batch_fails_before_store_construction(self):
        path = self.write_fixture([{"id": "invalid"}])
        with (
            patch("storage.seed.StaticParkingStore") as constructor,
            patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit) as error,
        ):
            main(["--file", str(path)])
        self.assertEqual(error.exception.code, 1)
        constructor.assert_not_called()

    def test_module_dry_run_works_from_another_directory_without_creating_data(self):
        env = {"PYTHONPATH": str(DEFAULT_SEED_FILE.parent.parent), "PATH": os.environ.get("PATH", "")}
        result = subprocess.run(
            [sys.executable, "-m", "storage.seed", "--dry-run"],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn(f"Validated {EXPECTED_SEED_COUNT} static records", result.stdout)
        self.assertEqual(list(self.root.iterdir()), [])
