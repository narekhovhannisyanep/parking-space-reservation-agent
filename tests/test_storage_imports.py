"""The SQLite-only recording process must not pay for public vector/model clients."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class StorageImportTests(unittest.TestCase):
    def check_import(self, code):
        project = Path(__file__).resolve().parent.parent
        with TemporaryDirectory() as directory:
            environment = {
                "PATH": os.getenv("PATH", ""),
                "HOME": directory,
                "PYTHONPATH": str(project),
                "LANGSMITH_TRACING": "false",
                "LANGCHAIN_TRACING_V2": "false",
                "PARKING_SQLITE_PATH": str(Path(directory) / "ledger.sqlite3"),
                "PARKING_CHROMA_DIRECTORY": str(Path(directory) / "chroma"),
            }
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=directory,
                env=environment,
                text=True,
                capture_output=True,
                timeout=45,
                check=False,
            )
            self.assertEqual(
                result.returncode, 0, "Offline import check failed; no provider credentials were supplied."
            )
            self.assertFalse(Path(environment["PARKING_SQLITE_PATH"]).exists())
            self.assertFalse(Path(environment["PARKING_CHROMA_DIRECTORY"]).exists())
            return json.loads(result.stdout)

    def test_recording_import_keeps_vector_and_provider_clients_unloaded(self):
        result = self.check_import(
            "import json, sys; import recording_server; "
            "print(json.dumps({name: name in sys.modules for name in ('chromadb', 'langchain_openai', 'presidio_analyzer', 'spacy')}))"
        )
        self.assertFalse(any(result.values()))

    def test_public_static_export_still_resolves_to_the_original_class(self):
        result = self.check_import(
            "import json, sys, storage; before = 'chromadb' in sys.modules; "
            "from storage import StaticParkingStore; from storage.static import StaticParkingStore as implementation; "
            "print(json.dumps({'eager': before, 'same': StaticParkingStore is implementation, 'cached': storage.StaticParkingStore is implementation}))"
        )
        self.assertFalse(result["eager"])
        self.assertTrue(result["same"] and result["cached"])


if __name__ == "__main__":
    unittest.main()
