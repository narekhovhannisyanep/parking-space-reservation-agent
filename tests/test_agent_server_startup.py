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
                "PYTHONPATH": os.pathsep.join([str(project), str(project / "tests")]),
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
                    from administrator_agent import ReviewWording
                    from langchain_core.runnables import RunnableLambda
                    from reservation_collection import FieldUpdate, ReservationExtraction
                    from storage import StaticParkingStore
                    from starlette.testclient import TestClient
                    from langgraph_api.server import app as server_app
                    from langgraph_api.graph import GRAPHS
                    from test_static_storage import KeywordEmbeddings

                    app.parking_agent.static_store = StaticParkingStore(
                        app.parking_agent.config.storage,
                        embeddings=KeywordEmbeddings(),
                        privacy_guard=app.parking_agent.privacy_guard,
                    )
                    extraction = ReservationExtraction()
                    app.parking_agent._reservation_extractor = RunnableLambda(lambda _: extraction)
                    review_calls = []

                    def compose_review(_):
                        review_calls.append(True)
                        return ReviewWording(message="Please review this reservation request.")

                    app.parking_agent._administrator_composer = RunnableLambda(compose_review)

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

                        # Exercise the actual endpoints that supply Studio's interrupt form.
                        response = client.post("/threads", json={})
                        assert response.status_code == 200, response.text
                        thread_id = response.json()["thread_id"]
                        thread_path = f"/threads/{thread_id}"

                        def run(**body):
                            response = client.post(thread_path + "/runs/wait", json={
                                "assistant_id": "parking_space_reservation", **body
                            })
                            assert response.status_code == 200, response.text
                            assert "__error__" not in response.json(), response.text
                            return response.json()

                        def state():
                            response = client.get(thread_path + "/state")
                            assert response.status_code == 200, response.text
                            return response.json()

                        def turn(text):
                            return run(input={"messages": [{"role": "user", "content": text}]})

                        details = {
                            "first_name": "Demo", "surname": "Tester",
                            "car_number": "my bicycle", "duration_hours": "2",
                        }
                        extraction = ReservationExtraction(intent="reservation", updates=[
                            FieldUpdate.model_validate({"field": key, "value": value})
                            for key, value in details.items()
                        ])
                        turn("Reserve: " + "; ".join(details.values()))
                        extraction = ReservationExtraction()
                        turn(f"/confirm {state()['values']['reservation']['revision']}")
                        turn("/options")
                        turn("/submit " + state()["values"]["offered_space_ids"][0])
                        pending = state()["tasks"][0]["interrupts"][0]
                        schema = pending["response_schema"]
                        assert schema["required"] == ["decision"], schema
                        assert set(schema["properties"]) == {"decision"}, schema
                        assert "default" not in schema["properties"]["decision"], schema
                        assert "copy" not in pending["value"]["instructions"].lower()
                        request_id = pending["value"]["request_id"]
                        original = app.parking_agent.reservation_store.get(request_id, thread_id)
                        assert original is not None and original.status == "pending"

                        run(command={"resume": {pending["id"]: {}}})
                        retry = state()["tasks"][0]["interrupts"][0]
                        assert "validation_error" in retry["value"], retry
                        assert retry["response_schema"] == schema
                        assert app.parking_agent.reservation_store.get(request_id, thread_id) == original
                        result = run(command={"resume": {retry["id"]: {"decision": "approve"}}})
                        assert "approved" in result["messages"][-1]["content"], result
                        assert not state()["tasks"]
                        row = app.parking_agent.reservation_store.get(request_id, thread_id)
                        assert row is not None and row.status == "approved" and row.approval_time is not None
                        assert "Recorded successfully" in result["messages"][-1]["content"], result
                        assert state()["values"]["recording"]["status"] == "recorded"
                        files = list(app.parking_agent.config.recording.directory.glob("*.txt"))
                        assert len(files) == 1, files
                        assert files[0].read_text() == (
                            f"Demo Tester | my bicycle | {row.request.start.isoformat()} to "
                            f"{row.request.end.isoformat()} | {row.approval_time.isoformat()}\\n"
                        )
                        assert len(review_calls) == 1, review_calls
                        assert app.parking_agent._llm_client is None
                    assert not app.parking_agent.startup_ready
                    print("Agent Server startup, thread isolation and decision-only review passed.")
            """)
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=60, check=False
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Agent Server startup, thread isolation and decision-only review passed.", result.stdout)


if __name__ == "__main__":
    unittest.main()
