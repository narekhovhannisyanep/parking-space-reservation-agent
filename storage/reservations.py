"""Code-owned private SQLite holds/decisions, never public RAG evidence or file output."""

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from reservation_collection import ValidatedReservationRequest, clean_text
from storage.dynamic import DynamicParkingStore, snapshot_eligible_ids
from storage.spaces import SpaceFilters

type ReservationStatus = Literal["pending", "approved", "refused", "elapsed"]
type HumanDecision = Literal["approve", "refuse"]


class ReservationConflict(ValueError):
    """An unavailable space, stale request, or forbidden state transition (no private data)."""


def timestamp(value: datetime) -> str:
    if value.utcoffset() is None:
        raise ValueError("The application clock and reservation period must be timezone-aware.")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class ReservationRecord:
    request_id: str
    thread_id: str
    space_id: str
    request: ValidatedReservationRequest
    status: ReservationStatus
    submitted_at: datetime
    decision_at: datetime | None
    approval_time: datetime | None

    def checkpoint(self) -> dict:
        """Only identity/status: private canonical fields are read from SQLite when needed."""
        return {"request_id": self.request_id, "status": self.status, "space_id": self.space_id}


class ReservationStore:
    """Shares the dynamic database lifecycle, but never its public CRUD/evidence API.

    The first write acquires SQLite's write lock BEFORE any reads. No transaction
    spans embeddings, model generation, or administrator review. Replayed nodes use
    the draft ID/revision as an idempotency key and cannot overwrite prior decisions.
    """

    def __init__(self, dynamic_store: DynamicParkingStore, *, clock: Callable[[], datetime] | None = None) -> None:
        self.dynamic_store = dynamic_store
        self.clock = clock if clock is not None else lambda: datetime.now(UTC)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ReservationRecord:
        return ReservationRecord(
            request_id=row["request_id"],
            thread_id=row["thread_id"],
            space_id=row["space_id"],
            request=ValidatedReservationRequest.model_validate_json(row["request_json"]),
            status=row["status"],
            submitted_at=datetime.fromisoformat(row["submitted_at"]),
            decision_at=datetime.fromisoformat(row["decision_at"]) if row["decision_at"] else None,
            approval_time=datetime.fromisoformat(row["approval_time"]) if row["approval_time"] else None,
        )

    def get(self, request_id: str, thread_id: str) -> ReservationRecord | None:
        with self.dynamic_store._connection() as connection:
            row = connection.execute(
                "SELECT * FROM parking_reservations WHERE request_id = ? AND thread_id = ?", (request_id, thread_id)
            ).fetchone()
        return self._from_row(row) if row else None

    def hold(
        self, request: ValidatedReservationRequest, space_id: str, thread_id: str, inventory_ids: Sequence[str]
    ) -> ReservationRecord:
        request = ValidatedReservationRequest.model_validate(request)
        thread_id = clean_text(thread_id)
        SpaceFilters(space_id=space_id).where(request.location_id)
        identifier = f"{request.location_id}:space:{space_id}"
        request_id = f"{request.draft_id}:{request.revision}"
        with self.dynamic_store._connection() as connection:
            connection.execute("UPDATE parking_reservations SET status = status WHERE 0")
            existing = connection.execute(
                "SELECT * FROM parking_reservations WHERE request_id = ?", (request_id,)
            ).fetchone()
            if existing:
                row = self._from_row(existing)
                if row.thread_id != thread_id or row.space_id != space_id or row.request != request:
                    raise ReservationConflict("This request identity has already been used with different details.")
                return row
            now = self.clock()
            submitted_at = timestamp(now)
            if request.start <= now:
                raise ReservationConflict(
                    "Reservation start must still be in the future. Correct and confirm the period."
                )
            facts = connection.execute(
                "SELECT * FROM parking_dynamic_facts WHERE location_id = ? AND kind = 'availability'",
                (request.location_id,),
            ).fetchall()
            snapshots = [self.dynamic_store._from_row(row) for row in facts]
            for fact in snapshots:
                self.dynamic_store.privacy_guard.check_public(fact.value)
            if identifier not in snapshot_eligible_ids(list(inventory_ids), snapshots, request.location_id):
                raise ReservationConflict("The selected space is no longer eligible. Choose a refreshed option.")
            overlap = connection.execute(
                "SELECT 1 FROM parking_reservations WHERE location_id = ? AND space_id = ? "
                "AND status IN ('pending', 'approved') AND start_at < ? AND end_at > ? LIMIT 1",
                (request.location_id, space_id, timestamp(request.end), timestamp(request.start)),
            ).fetchone()
            if overlap:
                raise ReservationConflict(
                    "The selected space is now held for an overlapping period. Choose a refreshed option."
                )
            connection.execute(
                "INSERT INTO parking_reservations (request_id, thread_id, location_id, space_id, request_json, "
                "start_at, end_at, status, submitted_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (
                    request_id,
                    thread_id,
                    request.location_id,
                    space_id,
                    request.model_dump_json(),
                    timestamp(request.start),
                    timestamp(request.end),
                    submitted_at,
                ),
            )
        return ReservationRecord(request_id, thread_id, space_id, request, "pending", now.astimezone(UTC), None, None)

    def decide(self, request_id: str, thread_id: str, decision: HumanDecision) -> ReservationRecord:
        if decision not in ("approve", "refuse"):
            raise ValueError("Administrator decision must be approve or refuse.")
        with self.dynamic_store._connection() as connection:
            connection.execute("UPDATE parking_reservations SET status = status WHERE 0")
            existing = connection.execute(
                "SELECT * FROM parking_reservations WHERE request_id = ? AND thread_id = ?", (request_id, thread_id)
            ).fetchone()
            if existing is None:
                raise ReservationConflict(
                    "The request is unavailable, possibly after the startup reset. Start a new request."
                )
            row = self._from_row(existing)
            if row.status != "pending":
                if row.status == {"approve": "approved", "refuse": "refused"}[decision] or row.status == "elapsed":
                    return row
                raise ReservationConflict("A completed administrator decision cannot be changed.")
            now = self.clock()
            decision_at = timestamp(now)
            status: ReservationStatus = "refused" if decision == "refuse" else "approved"
            if decision == "approve" and row.request.start <= now:
                status = "elapsed"
            connection.execute(
                "UPDATE parking_reservations SET status = ?, decision_at = ?, approval_time = ? WHERE request_id = ?",
                (status, decision_at, decision_at if status == "approved" else None, request_id),
            )
            updated = connection.execute(
                "SELECT * FROM parking_reservations WHERE request_id = ?", (request_id,)
            ).fetchone()
            assert updated is not None
            return self._from_row(updated)

    def blocked_spaces(self, location_id: str, start: datetime | None = None, end: datetime | None = None) -> list[str]:
        """Return IDs only. Unknown period checks current occupancy, not all future holds."""
        SpaceFilters().where(location_id)
        if (start is None) != (end is None):
            raise ValueError("Provide both period boundaries or neither.")
        now = timestamp(self.clock())
        if start is not None and end is not None:
            if end <= start:
                raise ValueError("Reservation end must be after start.")
            clause, parameters = "start_at < ? AND end_at > ?", (timestamp(end), timestamp(start))
        else:
            clause, parameters = "start_at <= ? AND end_at > ?", (now, now)
        with self.dynamic_store._connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT space_id FROM parking_reservations WHERE location_id = ? "
                "AND status IN ('pending', 'approved') AND end_at > ? AND " + clause + " ORDER BY space_id",
                (location_id, now, *parameters),
            ).fetchall()
        return [row["space_id"] for row in rows]
