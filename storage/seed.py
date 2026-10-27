"""Explicit trusted static-data ingestion: python -m storage.seed [--dry-run]."""

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document

from storage.config import StorageConfig
from storage.static import StaticParkingStore

DEFAULT_SEED_FILE = Path(__file__).with_name("static_examples.json")


def load_documents(path: Path = DEFAULT_SEED_FILE) -> list[Document]:
    """Load reviewed JSON records and reject the entire invalid batch before ingestion.

    Validation checks structure, not PII or source authenticity. Custom files must be
    reviewed by a trusted operator; never use chat or reservation exports as input.
    """
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Seed data must be a nonempty JSON array.")
    documents = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or set(record) != {"id", "page_content", "metadata"}:
            raise ValueError(f"Seed record {index} must contain only id, page_content, and metadata.")
        if (
            not isinstance(record["id"], str)
            or not isinstance(record["page_content"], str)
            or not isinstance(record["metadata"], dict)
        ):
            # Malformed external JSON is a validation failure, not Python API misuse.
            raise ValueError(f"Seed record {index} requires string id/text and an object metadata.")  # noqa: TRY004
        documents.append(Document(id=record["id"], page_content=record["page_content"], metadata=record["metadata"]))
    StaticParkingStore.validate_documents(documents)
    return documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate/upsert trusted public static parking JSON records.")
    parser.add_argument(
        "--file", type=Path, default=DEFAULT_SEED_FILE, help="Reviewed JSON file; defaults to demo records."
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only; no database, credentials, or API calls.")
    args = parser.parse_args(argv)
    try:
        documents = load_documents(args.file)
        locations = ", ".join(sorted({document.metadata["location_id"] for document in documents}))
        if args.dry_run:
            print(f"Validated {len(documents)} static records for: {locations}. No data written.")
            return 0
        config = StorageConfig.from_environment()
        store = StaticParkingStore(config)
        ids = store.upsert_documents(documents)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Static seeding failed: {error}\n")
    print(f"Upserted {len(ids)} static records for: {locations} into collection '{config.chroma_collection}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
