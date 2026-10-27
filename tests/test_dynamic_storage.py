"""Real SQLite CRUD, isolation, concurrency, and transaction checks, without seed data."""

import json
import sqlite3
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from storage import DynamicParkingStore, StorageConfig
from storage.dynamic import FactKind, JSONValue


class DynamicStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = StorageConfig(sqlite_path=Path(self.directory.name) / "nested/dynamic.sqlite3")
        self.store = DynamicParkingStore(self.config)

    def test_lazy_unseeded_database_and_missing_facts(self):
        self.assertFalse(self.config.sqlite_path.exists())
        self.assertIsNone(self.store.get_fact("north", "availability"))
        self.assertTrue(self.config.sqlite_path.exists())
        self.assertEqual(self.store.list_facts("north"), [])
        self.assertFalse(self.store.delete_fact("north", "availability"))

    def test_all_fact_kinds_round_trip_with_utc_timestamps(self):
        values: dict[FactKind, JSONValue] = {
            "working_hours": {"monday": ["08:00", "18:00"], "timezone": "Asia/Yerevan"},
            "prices": {"currency": "AMD", "hourly": 100.5, "discount": None},
            "availability": {"free": 0, "open": True},
        }
        for kind, value in values.items():
            written = self.store.upsert_fact("north", kind, value)
            self.assertEqual(self.store.get_fact("north", kind), written)
            self.assertEqual(written.value, value)
            self.assertEqual(written.updated_at.tzinfo, UTC)
        self.assertEqual([fact.kind for fact in self.store.list_facts("north")], sorted(values))

    def test_upsert_replaces_only_the_target_fact_and_delete_is_explicit(self):
        self.store.upsert_fact("north", "prices", {"hourly": 1})
        self.store.upsert_fact("south", "prices", {"hourly": 2})
        replacement = self.store.upsert_fact("north", "prices", {"hourly": 3})
        self.assertEqual(self.store.list_facts("north"), [replacement])
        south_fact = self.store.get_fact("south", "prices")
        assert south_fact is not None
        self.assertEqual(south_fact.value, {"hourly": 2})
        self.assertTrue(self.store.delete_fact("north", "prices"))
        self.assertFalse(self.store.delete_fact("north", "prices"))
        self.assertIsNotNone(self.store.get_fact("south", "prices"))

    def test_data_survives_a_new_instance_and_a_new_process(self):
        fact = self.store.upsert_fact("north", "availability", {"free": 7})
        self.assertEqual(DynamicParkingStore(self.config).get_fact("north", "availability"), fact)
        code = (
            "import json, sqlite3, sys; "
            "c = sqlite3.connect(sys.argv[1]); "
            "print(c.execute('SELECT value_json FROM parking_dynamic_facts').fetchone()[0]); c.close()"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(self.config.sqlite_path)], check=True, capture_output=True, text=True
        )
        self.assertEqual(json.loads(result.stdout), {"free": 7})

    def test_parameter_binding_prevents_sql_injection(self):
        location = "north'; DROP TABLE parking_dynamic_facts; --"
        fact = self.store.upsert_fact(location, "prices", "Quoted ' data | unicode Երևան")
        self.assertEqual(self.store.get_fact(location, "prices"), fact)
        self.assertEqual(self.store.list_facts("north"), [])
        self.assertTrue(self.store.delete_fact(location, "prices"))
        self.assertEqual(self.store.list_facts(location), [])

    def test_invalid_keys_and_values_do_not_create_a_database(self):
        invalid: list[tuple[str, Any, Any]] = [
            (" ", "prices", 1),
            ("north", "reservation", {}),
            ("north", "availability", float("nan")),
            ("north", "prices", float("inf")),
            ("north", "prices", {1: "converted key"}),
            ("north", "prices", {"nested": (1, 2)}),
        ]
        for location, kind, value in invalid:
            with self.subTest(location=location, kind=kind, value=value), self.assertRaises((ValueError, TypeError)):
                self.store.upsert_fact(location, kind, value)
        self.assertFalse(self.config.sqlite_path.exists())

    def test_circular_json_is_rejected_before_database_initialization(self):
        value: list[JSONValue] = []
        value.append(value)
        with self.assertRaisesRegex(ValueError, "Circular reference"):
            self.store.upsert_fact("north", "availability", value)
        self.assertFalse(self.config.sqlite_path.exists())

    def test_returned_json_does_not_share_mutable_state(self):
        nested: list[JSONValue] = [1]
        value: dict[str, JSONValue] = {"nested": nested}
        result = self.store.upsert_fact("north", "availability", value)
        nested.append(2)
        assert isinstance(result.value, dict)
        returned_nested = result.value["nested"]
        assert isinstance(returned_nested, list)
        returned_nested.append(3)
        stored = self.store.get_fact("north", "availability")
        assert stored is not None
        self.assertEqual(stored.value, {"nested": [1]})

    def test_connections_close_and_failed_transactions_roll_back(self):
        with self.store._connection() as connection:
            pass
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
        with self.assertRaisesRegex(RuntimeError, "rollback"), self.store._connection() as failed_connection:
            failed_connection.execute(
                "INSERT INTO parking_dynamic_facts VALUES (?, ?, ?, ?)",
                ("north", "prices", "1", "2026-01-01T00:00:00+00:00"),
            )
            raise RuntimeError("rollback")
        self.assertEqual(self.store.list_facts("north"), [])
        with self.assertRaises(sqlite3.ProgrammingError):
            failed_connection.execute("SELECT 1")

    def test_concurrent_workers_and_instances_have_independent_connections(self):
        stores = [self.store, DynamicParkingStore(self.config)]

        def write(index):
            store = stores[index % 2]
            location = f"location-{index}"
            store.upsert_fact(location, "availability", {"free": index})
            fact = store.get_fact(location, "availability")
            assert fact is not None
            return fact.value

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(write, range(32)))
        self.assertEqual(results, [{"free": index} for index in range(32)])

    def test_initialization_failure_can_be_retried(self):
        parent = self.config.sqlite_path.parent
        parent.parent.mkdir(parents=True, exist_ok=True)
        parent.touch()
        with self.assertRaises(FileExistsError):
            self.store.list_facts("north")
        self.assertFalse(self.store._initialized)
        parent.unlink()
        self.assertEqual(self.store.list_facts("north"), [])

    def test_concurrent_upserts_to_the_same_key_leave_one_complete_snapshot(self):
        stores = [self.store, DynamicParkingStore(self.config)]

        def write(index):
            return stores[index % 2].upsert_fact("north", "availability", {"free": index, "capacity": index + 1})

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(write, range(32)))
        final = self.store.get_fact("north", "availability")
        self.assertIn(final, results)
        self.assertEqual(len(self.store.list_facts("north")), 1)
