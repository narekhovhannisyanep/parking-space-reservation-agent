"""Explicit reviewed space-inventory ingestion: python -m storage.seed [--dry-run]."""

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document

from storage.config import StorageConfig
from storage.spaces import ParkingSpace
from storage.static import StaticParkingStore

DEFAULT_SEED_FILE = Path(__file__).with_name("parking_spaces.json")


def load_spaces(path: Path = DEFAULT_SEED_FILE) -> list[ParkingSpace]:
    """Load structured space records; reject the entire invalid batch before ingestion.

    Validation checks structure, not PII or source authenticity. Custom files must be
    reviewed by a trusted operator; never use chat or reservation exports as input.
    """
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Seed data must be a nonempty JSON array.")
    spaces = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            # Malformed external JSON is a data-validation failure, not Python API misuse.
            raise ValueError(f"Seed record {index} must be a space inventory object.")  # noqa: TRY004
        spaces.append(ParkingSpace.from_record(record))
    StaticParkingStore.validate_documents([space.to_document() for space in spaces])
    return spaces


def load_documents(path: Path = DEFAULT_SEED_FILE) -> list[Document]:
    """Derive descriptions and filtering metadata from the validated inventory."""
    return [space.to_document() for space in load_spaces(path)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate/upsert reviewed public parking-space inventory.")
    parser.add_argument(
        "--file", type=Path, default=DEFAULT_SEED_FILE, help="Reviewed inventory JSON; defaults to mock spaces."
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only; no database, credentials, or API calls.")
    args = parser.parse_args(argv)
    try:
        documents = load_documents(args.file)
        locations = ", ".join(sorted({document.metadata["location_id"] for document in documents}))
        if args.dry_run:
            print(f"Validated {len(documents)} parking spaces for: {locations}. No data written.")
            return 0
        config = StorageConfig.from_environment()
        store = StaticParkingStore(config)
        ids = store.upsert_documents(documents)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Space inventory seeding failed: {error}\n")
    print(f"Upserted {len(ids)} parking spaces for: {locations} into collection '{config.chroma_collection}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
