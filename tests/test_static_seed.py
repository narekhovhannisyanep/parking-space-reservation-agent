"""Seed validation and real Chroma ingestion with offline embeddings only."""

import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from test_static_storage import KeywordEmbeddings, document

from storage import ParkingSpace, SpaceFilters, StaticParkingStore, StorageConfig
from storage.seed import DEFAULT_SEED_FILE, load_documents, load_spaces, main

EXPECTED_SEED_COUNT = 20


class StaticSeedTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_fixture(self, records):
        path = self.root / "seed.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        return path

    def test_default_fixture_has_twenty_unique_spaces_and_no_workflow_records(self):
        documents = load_documents()
        self.assertEqual(len(documents), EXPECTED_SEED_COUNT)
        self.assertEqual(len({doc.id for doc in documents}), EXPECTED_SEED_COUNT)
        self.assertEqual(len({doc.page_content for doc in documents}), EXPECTED_SEED_COUNT)
        self.assertEqual({doc.metadata["location_id"] for doc in documents}, {"demo-central"})
        for doc in documents:
            assert doc.id is not None
            self.assertTrue(doc.id.startswith("demo-central:"))
            self.assertNotIn("synthetic", doc.page_content.lower())
            self.assertNotIn("fictional", doc.page_content.lower())
            self.assertTrue(doc.page_content.strip())
            self.assertLess(len(doc.page_content), 500)
            self.assertEqual(doc.metadata["source"], "mock-space-inventory-v2")
            self.assertEqual(doc.metadata["record_type"], "parking_space")
            ParkingSpace.validate_document(doc)
            for forbidden in ("topic", "status", "available", "hold_id", "car_number", "request_id", "price"):
                self.assertNotIn(forbidden, doc.metadata)

    def test_mock_inventory_is_minimal_and_vehicle_agnostic(self):
        spaces = load_spaces()
        self.assertEqual([space.to_document() for space in spaces], load_documents())
        for space in spaces:
            self.assertIn("Accepts any vehicle", space.to_document().page_content)
            self.assertEqual(set(vars(space)), {"space_id", "location_id", "label", "description", "source"})

    def test_seeded_inventory_filters_real_chroma_without_provider_calls(self):
        store = StaticParkingStore(StorageConfig(chroma_directory=self.root / "chroma"), embeddings=KeywordEmbeddings())
        store.upsert_documents(load_documents())
        results = store.search("near the lift", location_id="demo-central", filters=SpaceFilters(space_id="G-E02"))
        self.assertEqual([doc.metadata["space_id"] for doc in results], ["G-E02"])

    def test_loading_returns_independent_documents_and_does_not_touch_storage(self):
        with patch("storage.static.Chroma") as chroma, patch("storage.static.AzureOpenAIEmbeddings") as embeddings:
            first = load_documents()
            first[0].metadata["location_id"] = "changed"
            self.assertEqual(load_documents()[0].metadata["location_id"], "demo-central")
        chroma.assert_not_called()
        embeddings.assert_not_called()

    def test_invalid_shapes_fields_and_duplicate_space_ids_are_rejected(self):
        valid = json.loads(DEFAULT_SEED_FILE.read_text(encoding="utf-8"))[0]
        bad_records = [None, {}, [], [None], [{"id": "one"}]]
        for field, value in (
            ("space_id", 1),
            ("space_id", " "),
            ("description", None),
            ("description", " "),
            ("vehicle_type", "bus"),
            ("ev_charging", "yes"),
            ("width_m", -1),
            ("length_m", True),
            ("max_height_m", float("inf")),
        ):
            bad_records.append([{**valid, field: value}])
        for field in ("source", "space_id", "label", "location_id", "description"):
            missing = valid.copy()
            del missing[field]
            bad_records.append([missing])
        for field in ("car_number", "status", "available", "hold_id", "topic"):
            bad_records.append([valid, {**valid, "space_id": "second", field: "not static"}])
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
        records.append({**records[0], "space_id": "additional-space", "label": "Additional"})
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
        self.assertIn(f"Validated {EXPECTED_SEED_COUNT} parking spaces", output.getvalue())
        self.assertIn("No data written", output.getvalue())

    def test_cli_seed_is_repeatable_preserves_other_records_and_persists(self):
        config = StorageConfig(chroma_directory=self.root / "chroma", sqlite_path=self.root / "dynamic.sqlite3")
        store = StaticParkingStore(config, embeddings=KeywordEmbeddings())
        unrelated = document(location="other")
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
        self.assertIn(f"Upserted {EXPECTED_SEED_COUNT} parking spaces", output.getvalue())
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
        self.assertIn(f"Validated {EXPECTED_SEED_COUNT} parking spaces", result.stdout)
        self.assertEqual(list(self.root.iterdir()), [])
