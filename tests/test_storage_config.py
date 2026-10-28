"""Storage configuration stays immutable and does not depend on the working directory."""

import os
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest.mock import patch

from commons.constants import DIAL_TEXT_EMBEDDING_3_SMALL, TEXT_EMBEDDING_3_SMALL
from storage.config import PROJECT_ROOT, StorageConfig


class StorageConfigTests(unittest.TestCase):
    def test_empty_environment_uses_project_relative_persistent_defaults(self):
        for env in ({}, {"PARKING_SQLITE_PATH": "", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": ""}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                config = StorageConfig.from_environment()
                self.assertEqual(config.chroma_directory, PROJECT_ROOT / ".data/chroma")
                self.assertEqual(config.sqlite_path, PROJECT_ROOT / ".data/parking.sqlite3")
                self.assertEqual(config.chroma_collection, "parking_spaces_v2")
                self.assertEqual(config.embedding_deployment, DIAL_TEXT_EMBEDDING_3_SMALL)
                self.assertEqual(config.embedding_model, TEXT_EMBEDDING_3_SMALL)

    def test_environment_overrides_are_used(self):
        env = {
            "PARKING_CHROMA_DIRECTORY": "other/chroma",
            "PARKING_CHROMA_COLLECTION": "public_facts",
            "PARKING_SQLITE_PATH": "other/dynamic.sqlite3",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "embedding-deployment",
            "AZURE_OPENAI_EMBEDDING_MODEL": "embedding-model",
        }
        with patch.dict(os.environ, env, clear=True):
            config = StorageConfig.from_environment()
        self.assertEqual(config.chroma_directory, PROJECT_ROOT / "other/chroma")
        self.assertEqual(config.sqlite_path, PROJECT_ROOT / "other/dynamic.sqlite3")
        self.assertEqual(config.chroma_collection, "public_facts")
        self.assertEqual(config.embedding_deployment, "embedding-deployment")
        self.assertEqual(config.embedding_model, "embedding-model")

    def test_settings_are_immutable(self):
        for name in ("sqlite_path", "chroma_directory", "chroma_collection", "embedding_model", "embedding_deployment"):
            with self.subTest(name=name), self.assertRaises(FrozenInstanceError):
                setattr(StorageConfig(), name, None)

    def test_invalid_settings_are_rejected(self):
        cases: list[dict[str, Any]] = [
            {"chroma_collection": "ab"},
            {"chroma_collection": "bad collection"},
            {"chroma_collection": "-bad"},
            {"chroma_collection": "x" * 513},
            {"sqlite_path": Path(":memory:")},
            {"sqlite_path": Path(".")},
            {"embedding_deployment": " "},
            {"embedding_model": ""},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StorageConfig(**kwargs)
