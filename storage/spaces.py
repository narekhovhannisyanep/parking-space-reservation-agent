"""Minimal public space inventory, independent of databases and vehicle categories."""

import re
from dataclasses import asdict, dataclass, fields
from typing import Any

from langchain_core.documents import Document

SPACE_SCHEMA_VERSION = 2


def _text(name: str, value: str, *, identifier: bool = False, limit: int = 200) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be trimmed, nonblank text of at most {limit} characters.")
    if identifier and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", value):
        raise ValueError(f"{name} must be a 1-64 character alphanumeric/dash/underscore identifier.")


@dataclass(frozen=True)
class ParkingSpace:
    """One identified space, assumed to fit any vehicle in this project.

    No vehicle-fit restrictions, facility flags, availability, or customer data.
    """

    space_id: str
    location_id: str
    label: str
    description: str
    source: str

    def __post_init__(self) -> None:
        for name in ("space_id", "location_id"):
            _text(name, getattr(self, name), identifier=True)
        for name in ("label", "source"):
            _text(name, getattr(self, name))
        _text("description", self.description, limit=400)

    @property
    def document_id(self) -> str:
        return f"{self.location_id}:space:{self.space_id}"

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ParkingSpace":
        """Reject unknown/missing fields instead of silently ingesting private data."""
        if set(record) != {field.name for field in fields(cls)}:
            raise ValueError("Space records must contain exactly the documented inventory fields.")
        return cls(**record)

    def to_document(self) -> Document:
        """Generate searchable text and scalar filtering metadata from the same facts."""
        metadata = {name: value for name, value in asdict(self).items() if name != "description"}
        metadata.update(record_type="parking_space", schema_version=SPACE_SCHEMA_VERSION)
        text = (
            f"Space {self.label} ({self.space_id}) at {self.location_id}. "
            f"Accepts any vehicle under this project's assumption. {self.description}"
        )
        return Document(id=self.document_id, page_content=text, metadata=metadata)

    @classmethod
    def validate_document(cls, document: Document) -> None:
        """Validate metadata/identity; text review remains a trusted ingestion duty."""
        metadata = document.metadata.copy()
        if metadata.pop("record_type", None) != "parking_space":
            raise ValueError("Static documents must represent parking spaces, not guide/workflow records.")
        version = metadata.pop("schema_version", None)
        if type(version) is not int or version != SPACE_SCHEMA_VERSION:
            raise ValueError("Unsupported space metadata schema version.")
        if "description" in metadata:
            raise ValueError("Descriptions belong in document text, not filtering metadata.")
        space = cls.from_record({**metadata, "description": "Validated document metadata."})
        if document.id != space.document_id or not document.page_content.strip():
            raise ValueError("Space documents require the canonical location/space ID and nonblank text.")


@dataclass(frozen=True)
class SpaceFilters:
    """Optional exact space lookup within mandatory location/schema scope."""

    space_id: str | None = None

    def __post_init__(self) -> None:
        if self.space_id is not None:
            _text("space_id", self.space_id, identifier=True)

    def where(self, location_id: str) -> dict[str, Any]:
        """Always constrain the location and inventory schema, including on empty filters."""
        _text("location_id", location_id, identifier=True)
        clauses: list[dict[str, Any]] = [
            {"location_id": location_id},
            {"record_type": "parking_space"},
            {"schema_version": SPACE_SCHEMA_VERSION},
        ]
        if self.space_id is not None:
            clauses.append({"space_id": self.space_id})
        return {"$and": clauses}
