"""Private hold/decision contracts, using disposable SQLite and an injected clock."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from reservation_collection import ValidatedReservationRequest
from storage import DynamicParkingStore, StorageConfig
from storage.dynamic import JSONValue
from storage.reservations import ReservationConflict, ReservationStore

NOW = datetime(2030, 1, 1, tzinfo=UTC)
START = NOW + timedelta(days=1)
END = START + timedelta(hours=2)
IDS = ["demo-central:space:G-A01", "demo-central:space:G-A02"]


def request(identifier="draft-1", start=START, end=END, **changes):
    return ValidatedReservationRequest.model_validate(
        dict(
            draft_id=identifier,
            revision=1,
            location_id="demo-central",
            first_name="Demo",
            surname="Tester",
            car_number="my bicycle",
            start=start,
            end=end,
            timezone="UTC",
            **changes,
        )
    )


class ReservationStorageTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.config = StorageConfig(chroma_directory=root / "chroma", sqlite_path=root / "parking.sqlite3")
        self.dynamic = DynamicParkingStore(self.config)
        self.now = NOW
        self.store = ReservationStore(self.dynamic, clock=lambda: self.now)

    def hold(self, value=None, space="G-A01", thread="one", store=None):
        return (store or self.store).hold(value or request(), space, thread, IDS)

    def record(self, identifier):
        row = self.store.get(identifier, "one")
        assert row is not None
        return row

    def test_hold_is_private_persistent_and_reset_with_public_database(self):
        row = self.hold()
        self.assertEqual(row.status, "pending")
        self.assertIsNone(row.approval_time)
        self.assertEqual(self.dynamic.list_facts("demo-central"), [])
        reopened = ReservationStore(DynamicParkingStore(self.config), clock=lambda: self.now)
        self.assertEqual(reopened.get(row.request_id, "one"), row)
        self.assertIsNone(reopened.get(row.request_id, "other"))
        self.dynamic.reset()
        self.assertIsNone(self.store.get(row.request_id, "one"))

    def test_hold_retry_is_idempotent_but_identity_changes_are_rejected(self):
        row = self.hold()
        self.assertEqual(self.hold(), row)
        for value, space, thread in (
            (request(), "G-A02", "one"),
            (request(), "G-A01", "two"),
            (request(start=START + timedelta(minutes=1)), "G-A01", "one"),
        ):
            with self.subTest(space=space, thread=thread), self.assertRaises(ReservationConflict):
                self.hold(value, space, thread)

    def test_half_open_overlap_blocks_same_thread_and_other_threads(self):
        self.hold()
        for start, end in (
            (START, END),
            (START - timedelta(hours=1), START + timedelta(seconds=1)),
            (END - timedelta(seconds=1), END + timedelta(hours=1)),
            (START - timedelta(hours=1), END + timedelta(hours=1)),
        ):
            for thread in ("one", "two"):
                with self.subTest(start=start, thread=thread), self.assertRaises(ReservationConflict):
                    self.hold(request("other", start, end), thread=thread)
        self.hold(request("adjacent-after", END, END + timedelta(hours=1)))
        self.hold(request("adjacent-before", START - timedelta(hours=1), START))
        self.hold(request("different-space"), space="G-A02")

    def test_concurrent_overlapping_holds_have_exactly_one_winner_across_store_instances(self):
        self.dynamic._initialize()

        def submit(index):
            other = ReservationStore(DynamicParkingStore(self.config), clock=lambda: self.now)
            try:
                return self.hold(request(f"draft-{index}"), thread=f"thread-{index}", store=other)
            except ReservationConflict:
                return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            rows = list(executor.map(submit, range(16)))
        self.assertEqual(sum(row is not None for row in rows), 1)

    def test_public_overrides_and_inventory_are_rechecked_before_hold(self):
        values: list[JSONValue] = [False, {"G-A01": False}]
        for value in values:
            self.dynamic.upsert_fact("demo-central", "availability", value)
            with self.assertRaises(ReservationConflict):
                self.hold()
        self.dynamic.delete_fact("demo-central", "availability")
        with self.assertRaises(ReservationConflict):
            self.store.hold(request(), "NOT-OFFERED", "one", IDS)
        self.dynamic.upsert_fact("demo-central", "availability", {"G-A02": False})
        self.hold()

    def test_malformed_snapshot_fails_closed_without_inserting(self):
        self.dynamic.upsert_fact("demo-central", "availability", None)
        with self.assertRaises(RuntimeError):
            self.hold()
        self.assertIsNone(self.store.get("draft-1:1", "one"))

    def test_past_start_cannot_be_submitted(self):
        self.now = START
        with self.assertRaises(ReservationConflict):
            self.hold()

    def test_refusal_releases_hold_and_approval_keeps_it(self):
        row = self.hold()
        refused = self.store.decide(row.request_id, "one", "refuse")
        self.assertEqual(refused.status, "refused")
        self.assertIsNone(refused.approval_time)
        second = self.hold(request("second"), thread="two")
        approved = self.store.decide(second.request_id, "two", "approve")
        self.assertEqual(approved.status, "approved")
        self.assertEqual(approved.approval_time, NOW)
        with self.assertRaises(ReservationConflict):
            self.hold(request("third"))

    def test_decision_retries_preserve_timestamp_and_cannot_reverse_approval(self):
        row = self.hold()
        approved = self.store.decide(row.request_id, "one", "approve")
        self.now += timedelta(hours=1)
        self.assertEqual(self.store.decide(row.request_id, "one", "approve"), approved)
        with self.assertRaises(ReservationConflict):
            self.store.decide(row.request_id, "one", "refuse")

    def test_wrong_thread_missing_request_and_invalid_decisions_do_not_approve(self):
        row = self.hold()
        invalid: list[tuple] = [
            (row.request_id, "two", "approve"),
            ("missing", "one", "approve"),
            (row.request_id, "one", True),
            (row.request_id, "one", "yes"),
        ]
        for identifier, thread, decision in invalid:
            with (
                self.subTest(identifier=identifier, thread=thread, decision=decision),
                self.assertRaises((ValueError, ReservationConflict)),
            ):
                self.store.decide(identifier, thread, decision)
        self.assertEqual(self.record(row.request_id).status, "pending")

    def test_late_approval_becomes_elapsed_and_never_approved(self):
        row = self.hold()
        self.now = START
        result = self.store.decide(row.request_id, "one", "approve")
        self.assertEqual(result.status, "elapsed")
        self.assertIsNone(result.approval_time)
        self.assertEqual(self.store.blocked_spaces("demo-central", START, END), [])

    def test_period_suggestions_block_only_overlaps_without_exposing_private_details(self):
        self.hold()
        self.assertEqual(self.store.blocked_spaces("demo-central", START, END), ["G-A01"])
        self.assertEqual(self.store.blocked_spaces("demo-central", END, END + timedelta(hours=1)), [])
        self.assertEqual(self.store.blocked_spaces("other", START, END), [])
        self.assertEqual(self.store.blocked_spaces("demo-central"), [])
        self.now = START
        self.assertEqual(self.store.blocked_spaces("demo-central"), ["G-A01"])
        self.now = END
        self.assertEqual(self.store.blocked_spaces("demo-central"), [])
        self.assertEqual(self.record("draft-1:1").status, "pending")


if __name__ == "__main__":
    unittest.main()
