"""Non-secret storage settings; reading configuration does not create resources."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from commons.constants import DIAL_TEXT_EMBEDDING_3_SMALL, TEXT_EMBEDDING_3_SMALL

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class StorageConfig:
    """Local persistent stores; relative paths are anchored to the project root."""

    chroma_directory: Path = PROJECT_ROOT / ".data" / "chroma"
    chroma_collection: str = "parking_spaces_v2"
    sqlite_path: Path = PROJECT_ROOT / ".data" / "parking.sqlite3"
    embedding_deployment: str = DIAL_TEXT_EMBEDDING_3_SMALL
    embedding_model: str = TEXT_EMBEDDING_3_SMALL

    def __post_init__(self) -> None:
        for name in ("chroma_directory", "sqlite_path"):
            path = Path(getattr(self, name)).expanduser()
            if str(path) in (".", ":memory:"):
                raise ValueError("Storage paths must identify persistent disk locations.")
            object.__setattr__(self, name, (PROJECT_ROOT / path).resolve())
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]", self.chroma_collection):
            raise ValueError("Chroma collection must be 3-512 characters with alphanumeric endpoints.")
        for value in (self.embedding_deployment, self.embedding_model):
            if not value.strip():
                raise ValueError("Embedding model and deployment must not be blank.")

    @classmethod
    def from_environment(cls) -> "StorageConfig":
        return cls(
            chroma_directory=Path(os.getenv("PARKING_CHROMA_DIRECTORY") or ".data/chroma"),
            chroma_collection=os.getenv("PARKING_CHROMA_COLLECTION") or "parking_spaces_v2",
            sqlite_path=Path(os.getenv("PARKING_SQLITE_PATH") or ".data/parking.sqlite3"),
            embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or DIAL_TEXT_EMBEDDING_3_SMALL,
            embedding_model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL") or TEXT_EMBEDDING_3_SMALL,
        )
