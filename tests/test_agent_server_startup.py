"""Real Agent Server loader/lifespan smoke test, with isolated data and no outbound network."""

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from test_static_storage import KeywordEmbeddings

from storage import DynamicParkingStore, StaticParkingStore, StorageConfig
from storage.seed import load_documents


class AgentServerStartupTests(unittest.TestCase):
    def test_actual_server_shares_prepared_graph_and_does_not_reset_on_new_threads(self):
        project = Path(__file__).resolve().parent.parent
        with TemporaryDirectory() as directory:
            root = Path(directory)
            storage = StorageConfig(chroma_directory=root / "chroma", sqlite_path=root / "parking.sqlite3")
            StaticParkingStore(storage, embeddings=KeywordEmbeddings()).upsert_documents(load_documents())
            DynamicParkingStore(storage).upsert_fact("demo-central", "prices", 999)
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": directory,
                "PYTHONPATH": str(project),
                "PARKING_CHROMA_DIRECTORY": str(storage.chroma_directory),
                "PARKING_SQLITE_PATH": str(storage.sqlite_path),
                "PARKING_LOCATION_ID": "demo-central",
                "LANGGRAPH_RUNTIME_EDITION": "inmem",
                "LANGSMITH_LANGGRAPH_API_VARIANT": "local_dev",
                "DATABASE_URI": ":memory:",
                "REDIS_URI": "inmem",
                "MIGRATIONS_PATH": "__inmem",
                "LANGSERVE_GRAPHS": '{"parking_space_reservation":"app:agent"}',
                "LANGGRAPH_HTTP": '{"app":"app:webapp"}',
                "LANGSMITH_TRACING": "false",
                "LANGCHAIN_TRACING_V2": "false",
                "DISABLE_TRUSTSTORE": "true",
            }
            code = textwrap.dedent("""
                import socket
                from unittest.mock import patch

                def prohibit_network(*args, **kwargs):
                    raise AssertionError("Outbound network is forbidden in this smoke test")

                with patch.object(socket.socket, "connect", prohibit_network):
                    import app
                    from starlette.testclient import TestClient
                    from langgraph_api.server import app as server_app
                    from langgraph_api.graph import GRAPHS

                    assert not app.parking_agent.startup_ready
                    with TestClient(server_app) as client:
                        assert app.parking_agent.startup_ready
                        assert GRAPHS["parking_space_reservation"] is app.agent
                        assert app.parking_agent.baseline_context
                        assert app.parking_agent.dynamic_store.list_facts("demo-central") == []
                        app.parking_agent.dynamic_store.upsert_fact("demo-central", "prices", 7)
                        assert client.get("/ok").status_code == 200
                        for _ in range(2):
                            response = client.post("/runs/wait", json={
                                "assistant_id": "parking_space_reservation",
                                "input": {"messages": [{"role": "user", "content": "/new"}]}
                            })
                            assert response.status_code == 200, response.text
                            assert "messages" in response.json(), response.text
                            assert "Provide" in response.json()["messages"][-1]["content"]
                        fact = app.parking_agent.dynamic_store.get_fact("demo-central", "prices")
                        assert fact is not None and fact.value == 7
                    assert not app.parking_agent.startup_ready
                    print("Agent Server startup and thread isolation passed.")
            """)
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=60, check=False
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Agent Server startup and thread isolation passed.", result.stdout)


if __name__ == "__main__":
    unittest.main()
