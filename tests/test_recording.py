"""Approved-only recording and safe disposable output, using temporary local storage."""

import errno
import json
import os
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from test_reservation_storage import END, IDS, NOW, request

from recording import ApprovedReservationWriter, RecordingConfig, RecordingError, recording_filename
from storage import DynamicParkingStore, StorageConfig
from storage.reservations import ReservationStore


class RecordingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.storage = StorageConfig(chroma_directory=self.root / "chroma", sqlite_path=self.root / "parking.sqlite3")
        self.config = RecordingConfig.from_storage(self.storage)
        self.dynamic = DynamicParkingStore(self.storage)
        self.dynamic.reset()
        self.now = NOW
        self.ledger = ReservationStore(self.dynamic, clock=lambda: self.now)
        self.writer = ApprovedReservationWriter(self.config)
        self.writer.reset_output_directory()

    def approve(self, identifier="draft-1", **fields):
        row = self.ledger.hold(request(identifier).model_copy(update=fields), "G-A01", "one", IDS)
        return self.ledger.decide(row.request_id, "one", "approve")

    def files(self):
        return list(self.config.directory.glob("*.txt"))

    def test_only_approved_canonical_details_are_recorded_in_required_format(self):
        row = self.approve()
        result = self.writer.record(row.request_id, "one")
        self.assertEqual(result.status, "recorded")
        self.assertEqual(result.filename, recording_filename(row.request_id))
        self.assertEqual(
            self.files()[0].read_text(),
            f"Demo Tester | my bicycle | {NOW.isoformat()} to {END.isoformat()} | {NOW.isoformat()}\n",
        )
        self.assertEqual(self.ledger.get(row.request_id, "one"), row)
        self.assertEqual(self.dynamic.list_facts("demo-central"), [])
        self.assertNotIn("Demo", result.model_dump_json())
        self.assertNotIn("my bicycle", result.model_dump_json())
        self.assertEqual(self.config.directory.stat().st_mode & 0o777, 0o700)
        self.assertEqual(self.files()[0].stat().st_mode & 0o777, 0o600)

    def test_pending_refused_elapsed_missing_and_wrong_thread_never_write(self):
        row = self.ledger.hold(request(), "G-A01", "one", IDS)
        for identifier, thread in ((row.request_id, "one"), (row.request_id, "two"), ("missing", "one")):
            with self.subTest(identifier=identifier, thread=thread), self.assertRaises(RecordingError):
                self.writer.record(identifier, thread)
        self.ledger.decide(row.request_id, "one", "refuse")
        with self.assertRaises(RecordingError):
            self.writer.record(row.request_id, "one")
        ended = self.ledger.hold(request("ended"), "G-A01", "one", IDS)
        self.now = END
        self.ledger.decide(ended.request_id, "one", "approve")
        with self.assertRaises(RecordingError):
            self.writer.record(ended.request_id, "one")
        self.assertEqual(self.files(), [])

    def test_recording_after_end_keeps_original_period_and_approval_time(self):
        row = self.approve()
        self.now = END + timedelta(days=1)
        self.writer.record(row.request_id, "one")
        self.assertEqual(self.ledger.get(row.request_id, "one"), row)
        self.assertIn(NOW.isoformat(), self.files()[0].read_text())
        self.assertNotIn(self.now.isoformat(), self.files()[0].read_text())

    def test_unicode_and_nonplate_identifiers_are_preserved_not_used_as_paths(self):
        row = self.approve(first_name="Անի", surname="Գրիգորյան", car_number="../my bicycle 🚲")
        self.writer.record(row.request_id, "one")
        self.assertIn("Անի Գրիգորյան | ../my bicycle 🚲 |", self.files()[0].read_text())
        self.assertRegex(self.files()[0].name, r"^[a-f0-9]{64}\.txt$")

    def test_concurrent_replay_creates_exactly_one_complete_file(self):
        row = self.approve()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.writer.record(row.request_id, "one"), range(16)))
        self.assertTrue(all(result.status == "recorded" for result in results))
        self.assertEqual(len(self.files()), 1)
        self.assertEqual(self.files()[0].read_text().count("\n"), 1)
        self.assertEqual(len(list(self.config.directory.iterdir())), 1)

    def test_concurrent_distinct_approved_requests_have_independent_complete_files(self):
        rows = []
        for index in range(8):
            rows.append(self.approve(f"draft-{index}"))
            self.now += timedelta(hours=2)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda row: self.writer.record(row.request_id, "one"), rows))
        self.assertTrue(all(result.status == "recorded" for result in results))
        self.assertEqual(len(self.files()), len(rows))
        self.assertTrue(all(path.read_text().count("\n") == 1 for path in self.files()))

    def test_temporary_name_collision_never_deletes_another_writers_file(self):
        row = self.approve()
        existing = self.config.directory / ".recording-collision.tmp"
        existing.write_text("Another writer's temporary data")
        existing.chmod(0o600)
        with patch("recording.secrets.token_hex", return_value="collision"), self.assertRaises(RecordingError):
            self.writer.record(row.request_id, "one")
        self.assertTrue(existing.exists())
        self.assertEqual(existing.read_text(), "Another writer's temporary data")
        self.assertEqual(self.files(), [])

    def test_legacy_draft_ids_with_colons_and_maximum_length_are_not_path_interpreted(self):
        for index, identifier in enumerate((":draft:", "x" * 200)):
            with self.subTest(identifier=identifier):
                self.now = NOW + timedelta(hours=2 * index)
                row = self.approve(identifier)
                self.assertEqual(self.writer.record(row.request_id, "one").status, "recorded")

    def test_aware_approval_timestamp_is_rendered_in_utc_without_changing_its_instant(self):
        row = self.approve()
        offset = NOW.astimezone(timezone(timedelta(hours=3))).isoformat()
        with self.dynamic._connection() as connection:
            connection.execute(
                "UPDATE parking_reservations SET approval_time = ?, decision_at = ? WHERE request_id = ?",
                (offset, offset, row.request_id),
            )
        self.writer.record(row.request_id, "one")
        self.assertTrue(self.files()[0].read_text().endswith(f" | {NOW.isoformat()}\n"))

    def test_existing_conflicting_file_and_symlink_are_never_overwritten(self):
        row = self.approve()
        target = self.config.directory / recording_filename(row.request_id)
        target.write_text("Do not overwrite")
        target.chmod(0o600)
        with self.assertRaises(RecordingError) as error:
            self.writer.record(row.request_id, "one")
        self.assertEqual(error.exception.code, "conflict")
        self.assertEqual(target.read_text(), "Do not overwrite")
        target.unlink()
        outside = self.root / "outside.txt"
        outside.write_text("Outside sentinel")
        target.symlink_to(outside)
        with self.assertRaises(RecordingError):
            self.writer.record(row.request_id, "one")
        self.assertEqual(outside.read_text(), "Outside sentinel")

    def test_failure_before_publication_keeps_approval_and_no_partial_final_file(self):
        row = self.approve()
        with (
            patch("recording.os.link", side_effect=OSError(errno.ENOSPC, "private/path")),
            self.assertRaises(RecordingError) as error,
        ):
            self.writer.record(row.request_id, "one")
        self.assertFalse(error.exception.unconfirmed)
        self.assertFalse(error.exception.retryable)
        self.assertNotIn("private/path", str(error.exception))
        self.assertEqual(self.files(), [])
        self.assertEqual(list(self.config.directory.iterdir()), [])
        self.assertEqual(self.ledger.get(row.request_id, "one"), row)
        self.writer.record(row.request_id, "one")

    def test_failure_after_publication_is_unconfirmed_and_retry_is_idempotent(self):
        row = self.approve()
        fsync = os.fsync

        def fail_directory_sync(fd):
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError(errno.EIO, "private directory failure")
            return fsync(fd)

        with patch("recording.os.fsync", side_effect=fail_directory_sync), self.assertRaises(RecordingError) as error:
            self.writer.record(row.request_id, "one")
        self.assertTrue(error.exception.unconfirmed)
        self.assertEqual(len(self.files()), 1)
        self.assertEqual(self.writer.record(row.request_id, "one").status, "recorded")
        self.assertEqual(len(self.files()), 1)

    def test_corrupt_private_fields_cannot_inject_lines_or_change_canonical_period(self):
        row = self.approve()
        for field, value in (("car_number", "bad\nentry"), ("first_name", "bad | entry"), ("end", NOW.isoformat())):
            with self.subTest(field=field):
                data = row.request.model_dump(mode="json")
                data[field] = value
                with self.dynamic._connection() as connection:
                    connection.execute(
                        "UPDATE parking_reservations SET request_json = ? WHERE request_id = ?",
                        (json.dumps(data), row.request_id),
                    )
                with self.assertRaises(RecordingError):
                    self.writer.record(row.request_id, "one")
                self.assertEqual(self.files(), [])

    def test_read_only_writer_does_not_create_or_resurrect_a_database(self):
        self.storage.sqlite_path.unlink()
        with self.assertRaises(RecordingError):
            self.writer.record("draft-1:1", "one")
        self.assertFalse(self.storage.sqlite_path.exists())
        self.assertEqual(self.files(), [])

    def test_startup_cleanup_is_confined_and_rejects_symlink_or_overlapping_paths(self):
        self.config.directory.joinpath("old.txt").write_text("Disposable")
        sentinel = self.root / "keep.txt"
        sentinel.write_text("Preserve")
        self.storage.chroma_directory.mkdir()
        self.storage.chroma_directory.joinpath("keep.txt").write_text("Static")
        self.writer.reset_output_directory()
        self.assertEqual(list(self.config.directory.iterdir()), [])
        self.assertEqual(sentinel.read_text(), "Preserve")
        self.assertTrue(self.storage.sqlite_path.exists())
        self.assertEqual(self.storage.chroma_directory.joinpath("keep.txt").read_text(), "Static")
        for directory, database, chroma in (
            (self.root, self.storage.sqlite_path, self.storage.chroma_directory),
            (self.config.directory, self.config.directory / "ledger.sqlite3", self.storage.chroma_directory),
            (self.config.directory, self.storage.sqlite_path, self.config.directory / "chroma"),
            (
                self.storage.chroma_directory / "approved-reservations",
                self.storage.sqlite_path,
                self.storage.chroma_directory,
            ),
        ):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                replace(self.config, directory=directory, sqlite_path=database, chroma_directory=chroma)
        self.config.directory.rmdir()
        self.config.directory.symlink_to(self.storage.chroma_directory, target_is_directory=True)
        with self.assertRaises((ValueError, RecordingError)):
            self.writer.reset_output_directory()
        self.assertTrue(self.storage.chroma_directory.joinpath("keep.txt").exists())

    def test_readiness_requires_existing_schema_and_secure_output(self):
        self.writer.check_ready()
        self.config.directory.chmod(0o755)
        with self.assertRaises(RecordingError):
            self.writer.check_ready()
        self.config.directory.chmod(0o700)
        self.storage.sqlite_path.unlink()
        sqlite3.connect(self.storage.sqlite_path).close()
        with self.assertRaises(RecordingError):
            self.writer.check_ready()


if __name__ == "__main__":
    unittest.main()
