"""Private approved-only output. SQLite owns approval/allocation, never these files."""

import errno
import hashlib
import os
import secrets
import shutil
import sqlite3
import stat
from collections.abc import Generator
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator, model_validator

from reservation_collection import StrictModel, clean_text
from storage.reservations import ReservationRecord, ReservationStore

if TYPE_CHECKING:
    from storage.config import StorageConfig

type RecordingErrorCode = Literal[
    "invalid_request",
    "not_approved",
    "unavailable",
    "unsafe_path",
    "conflict",
    "storage_error",
    "transport_error",
    "invalid_response",
]


@dataclass(frozen=True)
class RecordingConfig:
    """A dedicated sibling of disposable SQLite; no chat-selected paths or secrets."""

    directory: Path
    sqlite_path: Path
    chroma_directory: Path

    def __post_init__(self) -> None:
        for field in ("directory", "sqlite_path", "chroma_directory"):
            object.__setattr__(self, field, Path(os.path.abspath(Path(getattr(self, field)).expanduser())))
        self.validate_paths()

    @classmethod
    def from_storage(cls, storage: "StorageConfig") -> "RecordingConfig":
        return cls(storage.sqlite_path.parent / "approved-reservations", storage.sqlite_path, storage.chroma_directory)

    def validate_paths(self) -> None:
        # Keep lexical paths until after the symlink check. resolve() alone could
        # turn an unsafe cleanup symlink into its innocent-looking target.
        if self.directory.name != "approved-reservations":
            raise ValueError("Recording output must use a dedicated approved-reservations directory.")
        for path in (self.directory, self.sqlite_path):
            if any(component.is_symlink() for component in (path, *path.parents)):
                raise ValueError("Recording storage paths must not contain symbolic links.")
        directory, database, chroma = (
            self.directory.resolve(),
            self.sqlite_path.resolve(),
            self.chroma_directory.resolve(),
        )
        if directory.is_relative_to(chroma) or chroma.is_relative_to(directory):
            raise ValueError("Disposable recording output must not overlap persistent Chroma storage.")
        if database.is_relative_to(directory) or directory.is_relative_to(database):
            raise ValueError("Recording output must not overlap the SQLite database.")


class RecordingInput(StrictModel):
    request_id: str = Field(min_length=1, max_length=512)
    thread_id: str

    @field_validator("request_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        # Draft IDs may be 200 characters before adding ':revision'. Hashing,
        # not interpreting the ID as a path, permits every existing draft ID.
        draft, separator, revision = value.rpartition(":")
        if separator:
            clean_text(draft)
            clean_text(revision)
        else:
            clean_text(value)
        if value != value.strip():
            raise ValueError("Request identity must not contain surrounding whitespace.")
        return value

    @field_validator("thread_id")
    @classmethod
    def validate_thread(cls, value: str) -> str:
        if value != clean_text(value):
            raise ValueError("Thread identity must not contain surrounding whitespace.")
        return value


def recording_filename(request_id: str) -> str:
    return hashlib.sha256(request_id.encode()).hexdigest() + ".txt"


class RecordingResult(StrictModel):
    """Only bounded identity/outcome data crosses MCP or enters graph checkpoints."""

    status: Literal["recorded", "failed", "unconfirmed"]
    request_id: str = Field(max_length=512)
    filename: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}\.txt$")
    error_code: RecordingErrorCode | None = None
    retryable: bool = False
    attempts: int = Field(default=1, ge=1, le=2)

    @model_validator(mode="after")
    def validate_outcome(self) -> "RecordingResult":
        if self.status == "recorded":
            if self.filename != recording_filename(self.request_id) or self.error_code is not None or self.retryable:
                raise ValueError("A recording receipt must match its request and have no error.")
        elif self.filename is not None or self.error_code is None:
            raise ValueError("Unsuccessful recording must have an error code, not a success receipt.")
        return self


class RecordingError(RuntimeError):
    """Safe error codes; never forward SQL, personal details, or filesystem exceptions."""

    def __init__(self, code: RecordingErrorCode, *, retryable: bool = False, unconfirmed: bool = False) -> None:
        super().__init__("Reservation recording failed. Check the local recording service.")
        self.code: RecordingErrorCode = code
        self.retryable = retryable
        self.unconfirmed = unconfirmed

    def result(self, request_id: str) -> RecordingResult:
        return RecordingResult(
            status="unconfirmed" if self.unconfirmed else "failed",
            request_id=request_id,
            error_code=self.code,
            retryable=self.retryable,
        )


def retryable_storage_error(error: OSError | sqlite3.Error) -> bool:
    if isinstance(error, sqlite3.Error):
        return getattr(error, "sqlite_errorcode", 0) & 0xFF in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
    return error.errno in {errno.EAGAIN, errno.EINTR, errno.EBUSY, errno.ETIMEDOUT}


class ApprovedReservationWriter:
    """Read-only ledger access and atomic no-overwrite publication on local POSIX storage."""

    def __init__(self, config: RecordingConfig) -> None:
        self.config = config

    @staticmethod
    def _directory_flags() -> int:
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise RecordingError("unsafe_path")
        return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW

    @contextmanager
    def _directory(self) -> Generator[int, None, None]:
        try:
            self.config.validate_paths()
            fd = os.open(self.config.directory, self._directory_flags())
        except (OSError, ValueError):
            raise RecordingError("unsafe_path") from None
        try:
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise RecordingError("unsafe_path")
            yield fd
        finally:
            os.close(fd)

    def reset_output_directory(self) -> None:
        """Server startup only. Never delete the parent, SQLite, or persistent Chroma."""
        self.config.validate_paths()
        if not shutil.rmtree.avoids_symlink_attacks:
            raise RecordingError("unsafe_path")
        self.config.directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.config.directory.parent, self._directory_flags())
        try:
            try:
                existing = os.stat(self.config.directory.name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if not stat.S_ISDIR(existing.st_mode) or existing.st_uid != os.getuid():
                    raise RecordingError("unsafe_path")
                shutil.rmtree(self.config.directory.name, dir_fd=fd)
            os.mkdir(self.config.directory.name, mode=0o700, dir_fd=fd)
            os.fsync(fd)
        finally:
            os.close(fd)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        self.config.validate_paths()
        # mode=ro must fail if the file is absent. Never use DynamicParkingStore's
        # connection helper here: it initializes a missing database/schema.
        with closing(
            sqlite3.connect(self.config.sqlite_path.as_uri() + "?mode=ro", uri=True, timeout=0.25)
        ) as connection:
            connection.row_factory = sqlite3.Row
            yield connection

    def check_ready(self) -> None:
        try:
            with self._directory(), self._connection() as connection:
                connection.execute(
                    "SELECT request_id, thread_id, request_json, status, approval_time, decision_at, "
                    "submitted_at, start_at, end_at, location_id, space_id FROM parking_reservations LIMIT 0"
                )
        except (OSError, sqlite3.Error):
            raise RecordingError("unavailable") from None

    def _approved_record(self, value: RecordingInput) -> ReservationRecord:
        try:
            with self._connection() as connection:
                raw = connection.execute(
                    "SELECT * FROM parking_reservations WHERE request_id = ? AND thread_id = ?",
                    (value.request_id, value.thread_id),
                ).fetchone()
        except (OSError, sqlite3.Error) as error:
            raise RecordingError("unavailable", retryable=retryable_storage_error(error)) from None
        if raw is None:
            raise RecordingError("unavailable")
        if raw["status"] != "approved":
            raise RecordingError("not_approved")
        try:
            row = ReservationStore._from_row(raw)
            if (
                row.request_id != f"{row.request.draft_id}:{row.request.revision}"
                or row.request.location_id != raw["location_id"]
                or row.request.start != datetime.fromisoformat(raw["start_at"])
                or row.request.end != datetime.fromisoformat(raw["end_at"])
                or row.submitted_at != row.request.start
                or row.approval_time is None
                or row.approval_time.utcoffset() is None
                or row.approval_time != row.decision_at
                or not row.request.start <= row.approval_time < row.request.end
            ):
                raise ValueError("Inconsistent approved record.")
        except (TypeError, ValueError):
            raise RecordingError("invalid_request") from None
        return row

    @staticmethod
    def _entry(row: ReservationRecord) -> bytes:
        assert row.approval_time is not None
        return (
            f"{row.request.first_name} {row.request.surname} | {row.request.car_number} | "
            f"{row.request.start.isoformat()} to {row.request.end.isoformat()} | {row.approval_time.astimezone(UTC).isoformat()}\n"
        ).encode()

    @staticmethod
    def _matches(fd: int, name: str, expected: bytes) -> bool:
        # O_NONBLOCK avoids hanging on a malicious FIFO before fstat rejects it.
        existing = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(existing, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise RecordingError("unsafe_path")
            return stream.read(len(expected) + 1) == expected

    def record(self, request_id: str, thread_id: str) -> RecordingResult:
        try:
            value = RecordingInput(request_id=request_id, thread_id=thread_id)
            self.config.validate_paths()
        except ValueError:
            raise RecordingError("invalid_request") from None
        row = self._approved_record(value)
        entry, name = self._entry(row), recording_filename(value.request_id)
        with self._directory() as directory:
            temporary: str | None = ".recording-" + secrets.token_hex(16) + ".tmp"
            owns_temporary = False
            published = False
            try:
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
                owns_temporary = True
                with os.fdopen(fd, "wb") as stream:
                    stream.write(entry)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    # link is atomic and fails if the destination already exists.
                    # The temporary and final names live on the same filesystem.
                    os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
                except FileExistsError:
                    if not self._matches(directory, name, entry):
                        raise RecordingError("conflict") from None
                published = True
                os.unlink(temporary, dir_fd=directory)
                temporary = None
                os.fsync(directory)
            except OSError as error:
                raise RecordingError(
                    "storage_error", retryable=retryable_storage_error(error), unconfirmed=published
                ) from None
            finally:
                if owns_temporary and temporary is not None:
                    with suppress(OSError):
                        os.unlink(temporary, dir_fd=directory)
        return RecordingResult(status="recorded", request_id=value.request_id, filename=name)
