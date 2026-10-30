"""SQLite snapshots of public dynamic facts, not an allocation or reservation ledger."""

import json
import sqlite3
from collections.abc import Generator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Literal

from guardrails import PrivacyGuard
from storage.config import StorageConfig

type FactKind = Literal["working_hours", "prices", "availability"]
type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]


@dataclass(frozen=True)
class DynamicFact:
    """A public snapshot with a UTC write timestamp, not a freshness guarantee."""

    location_id: str
    kind: FactKind
    value: JSONValue
    updated_at: datetime


class DynamicParkingStore:
    """One connection per operation, explicitly committed/closed, safe across graph workers.

    JSON payloads defer the domain schema until sources and parking topology are
    decided. Availability snapshots do not reserve capacity or authorize bookings.
    """

    def __init__(self, config: StorageConfig, *, privacy_guard: PrivacyGuard | None = None) -> None:
        self.config = config
        self.privacy_guard = privacy_guard if privacy_guard is not None else PrivacyGuard()
        self._initialized = False
        self._lock = Lock()

    def _initialize(self) -> None:
        with self._lock:
            if not self._initialized:
                self.config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
                with (
                    closing(sqlite3.connect(self.config.sqlite_path, timeout=30, autocommit=False)) as connection,
                    connection,
                ):
                    connection.execute(
                        """CREATE TABLE IF NOT EXISTS parking_dynamic_facts (
                            location_id TEXT NOT NULL CHECK(length(trim(location_id)) > 0),
                            kind TEXT NOT NULL CHECK(kind IN ('working_hours', 'prices', 'availability')),
                            value_json TEXT NOT NULL,
                            updated_at TEXT NOT NULL,
                            PRIMARY KEY (location_id, kind)
                        )"""
                    )
                self._initialized = True

    def reset(self) -> None:
        """Recreate the configured disposable database at server startup only.

        No store operation retains a connection. Call before accepting interactions,
        never while graph threads are running. Remove SQLite sidecars too, so an old
        WAL cannot restore data into the new database. Never delete the parent folder.
        """
        with self._lock:
            self._initialized = False
            for suffix in ("", "-wal", "-shm", "-journal"):
                self.config.sqlite_path.with_name(self.config.sqlite_path.name + suffix).unlink(missing_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        self._initialize()
        with closing(sqlite3.connect(self.config.sqlite_path, timeout=30, autocommit=False)) as connection:
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection

    @staticmethod
    def _validate_key(location_id: str, kind: FactKind) -> None:
        if not location_id.strip():
            raise ValueError("location_id must not be blank.")
        if kind not in ("working_hours", "prices", "availability"):
            raise ValueError("Unsupported dynamic parking fact kind.")

    @staticmethod
    def _validate_json(value: JSONValue) -> None:
        # Prevent silent conversion of non-JSON keys and Python tuples by json.dumps.
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings.")
                DynamicParkingStore._validate_json(item)
        elif isinstance(value, list):
            for item in value:
                DynamicParkingStore._validate_json(item)
        elif value is not None and type(value) not in (str, int, float, bool):
            raise ValueError("Dynamic values must be JSON-compatible.")

    @staticmethod
    def _from_row(row: sqlite3.Row) -> DynamicFact:
        return DynamicFact(
            location_id=row["location_id"],
            kind=row["kind"],
            value=json.loads(row["value_json"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def upsert_fact(self, location_id: str, kind: FactKind, value: JSONValue) -> DynamicFact:
        """Atomically replace one snapshot; no LLM-generated SQL or reservation writes."""
        self._validate_key(location_id, kind)
        # Let the encoder reject cycles before traversing nested values ourselves.
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        self._validate_json(value)
        self.privacy_guard.check_public(location_id)
        self.privacy_guard.check_public(value)
        with self._connection() as connection:
            # Acquire the write lock before timestamping to order concurrent writes.
            connection.execute("UPDATE parking_dynamic_facts SET updated_at = updated_at WHERE 0")
            updated_at = datetime.now(UTC)
            connection.execute(
                """INSERT INTO parking_dynamic_facts (location_id, kind, value_json, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(location_id, kind) DO UPDATE SET
                     value_json = excluded.value_json, updated_at = excluded.updated_at""",
                (location_id, kind, encoded, updated_at.isoformat()),
            )
        return DynamicFact(location_id, kind, json.loads(encoded), updated_at)

    def get_fact(self, location_id: str, kind: FactKind) -> DynamicFact | None:
        self._validate_key(location_id, kind)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM parking_dynamic_facts WHERE location_id = ? AND kind = ?", (location_id, kind)
            ).fetchone()
        return None if row is None else self._from_row(row)

    def list_facts(self, location_id: str) -> list[DynamicFact]:
        if not location_id.strip():
            raise ValueError("location_id must not be blank.")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM parking_dynamic_facts WHERE location_id = ? ORDER BY kind", (location_id,)
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def delete_fact(self, location_id: str, kind: FactKind) -> bool:
        self._validate_key(location_id, kind)
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM parking_dynamic_facts WHERE location_id = ? AND kind = ?", (location_id, kind)
            )
            return cursor.rowcount > 0
