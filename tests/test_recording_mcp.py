"""Real MCP protocol/stdio checks plus bounded client retries; no provider calls."""

import asyncio
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, PropertyMock, patch

import anyio
from mcp import Client
from mcp.server import Server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
from test_reservation_storage import IDS, NOW, request

from recording import (
    ApprovedReservationWriter,
    RecordingConfig,
    RecordingError,
    RecordingInput,
    RecordingResult,
    recording_filename,
)
from recording_client import MCPRecordingClient
from recording_server import TOOL_NAME, create_server
from storage import DynamicParkingStore, StorageConfig
from storage.reservations import ReservationStore


def in_process_recorder(storage):
    """Real MCP without per-fixture subprocess startup; stdio is tested separately."""
    config = RecordingConfig.from_storage(storage)
    return MCPRecordingClient(config, client_factory=lambda: Client(create_server(config)))


class MCPRecordingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.storage = StorageConfig(chroma_directory=root / "chroma", sqlite_path=root / "parking.sqlite3")
        self.config = RecordingConfig.from_storage(self.storage)
        self.dynamic = DynamicParkingStore(self.storage)
        self.dynamic.reset()
        self.ledger = ReservationStore(self.dynamic, clock=lambda: NOW)
        self.writer = ApprovedReservationWriter(self.config)
        self.writer.reset_output_directory()
        self.recorder = MCPRecordingClient(self.config)

    def approve(self):
        row = self.ledger.hold(request(), "G-A01", "one", IDS)
        return self.ledger.decide(row.request_id, "one", "approve")

    async def test_real_stdio_writes_and_replay_preserves_one_file(self):
        row = self.approve()
        result = await self.recorder.arecord(row.request_id, "one")
        replay = await self.recorder.arecord(row.request_id, "one")
        self.assertEqual(result.status, "recorded")
        self.assertEqual(replay.filename, result.filename)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(list(self.config.directory.glob("*.txt"))), 1)
        self.assertNotIn("Demo", result.model_dump_json())
        self.assertEqual(self.ledger.get(row.request_id, "one"), row)

    def test_real_stdio_sync_api_and_startup_check(self):
        self.recorder.check_ready()
        row = self.approve()
        self.assertEqual(self.recorder.record(row.request_id, "one").status, "recorded")

    async def test_server_advertises_only_narrow_schema_and_rejects_forged_authority(self):
        row = self.ledger.hold(request(), "G-A01", "one", IDS)
        async with Client(create_server(self.config)) as client:
            tools = (await client.list_tools()).tools
            self.assertEqual([tool.name for tool in tools], [TOOL_NAME])
            self.assertEqual(set(tools[0].input_schema["properties"]), {"request_id", "thread_id"})
            self.assertFalse(tools[0].input_schema["additionalProperties"])
            for args in (
                {"request_id": row.request_id, "thread_id": "one"},
                {"request_id": row.request_id, "thread_id": "one", "approved": True},
                {"request_id": row.request_id, "thread_id": "one", "path": "../outside.txt"},
                {"request_id": row.request_id, "thread_id": "one", "car_number": "secret-car"},
                {"request_id": row.request_id, "thread_id": "two"},
                {"request_id": True, "thread_id": "one"},
                {},
            ):
                with self.subTest(args=args):
                    result = await client.call_tool(TOOL_NAME, args)
                    self.assertTrue(result.is_error)
                    self.assertNotIn("secret-car", str(result.content))
            self.assertTrue((await client.call_tool("write_file", {})).is_error)
        self.assertEqual(list(self.config.directory.iterdir()), [])
        self.assertEqual(self.ledger.get(row.request_id, "one"), row)

    async def test_transient_failure_retries_once_and_exhaustion_is_unconfirmed(self):
        receipt = RecordingResult(status="recorded", request_id="draft-1:1", filename=recording_filename("draft-1:1"))
        with patch.object(
            self.recorder, "_call", AsyncMock(side_effect=[TimeoutError("private-token"), receipt])
        ) as call:
            result = await self.recorder.arecord("draft-1:1", "one")
        self.assertEqual(result.status, "recorded")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(call.await_count, 2)
        with patch.object(self.recorder, "_call", AsyncMock(side_effect=TimeoutError("private-token"))) as call:
            result = await self.recorder.arecord("draft-1:1", "one")
        self.assertEqual(result.status, "unconfirmed")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(call.await_count, 2)
        self.assertNotIn("private-token", result.model_dump_json())

    async def test_nonretryable_failure_and_cancellation_do_not_retry(self):
        failure = RecordingResult(status="failed", request_id="draft-1:1", error_code="not_approved")
        with patch.object(self.recorder, "_call", AsyncMock(return_value=failure)) as call:
            result = await self.recorder.arecord("draft-1:1", "one")
        self.assertEqual(result.status, "failed")
        self.assertEqual(call.await_count, 1)
        with (
            patch.object(self.recorder, "_call", AsyncMock(side_effect=asyncio.CancelledError)) as call,
            self.assertRaises(asyncio.CancelledError),
        ):
            await self.recorder.arecord("draft-1:1", "one")
        self.assertEqual(call.await_count, 1)
        with (
            patch.object(self.recorder, "_call", AsyncMock(side_effect=RuntimeError("programming error"))),
            self.assertRaisesRegex(RuntimeError, "programming error"),
        ):
            await self.recorder.arecord("draft-1:1", "one")

    def test_malformed_mismatched_and_error_flag_results_cannot_claim_success(self):
        valid = RecordingResult(status="recorded", request_id="draft-1:1", filename=recording_filename("draft-1:1"))
        for data, is_error in (
            (None, False),
            ({**valid.model_dump(), "request_id": "other:1"}, False),
            ({**valid.model_dump(), "filename": "../outside.txt"}, False),
            ({**valid.model_dump(), "extra": "private-secret"}, False),
            (valid.model_dump(), True),
        ):
            with self.subTest(data=data, is_error=is_error):
                raw = CallToolResult(
                    content=[TextContent(type="text", text="private-secret")],
                    structured_content=data,
                    is_error=is_error,
                )
                result = self.recorder.validate_result(raw, "draft-1:1")
                self.assertEqual(result.status, "unconfirmed")
                self.assertNotIn("private-secret", result.model_dump_json())

    def test_child_command_is_fixed_and_does_not_inherit_provider_credentials(self):
        with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "never-forward", "LANGSMITH_API_KEY": "never-forward"}):
            parameters = self.recorder.server_parameters
        self.assertTrue(Path(parameters.command).is_absolute())
        self.assertTrue(Path(parameters.args[0]).is_absolute())
        self.assertEqual(Path(parameters.args[0]).name, "recording_server.py")
        self.assertNotIn("AZURE_OPENAI_API_KEY", parameters.env or {})
        self.assertNotIn("LANGSMITH_API_KEY", parameters.env or {})
        self.assertNotIn("never-forward", str(parameters))

    def broken_server(self, *, bad_schema=False):
        async def list_tools(ctx, params):
            return ListToolsResult(
                tools=[
                    Tool(
                        name=TOOL_NAME,
                        input_schema={"type": "object"} if bad_schema else RecordingInput.model_json_schema(),
                        output_schema=RecordingResult.model_json_schema(),
                    )
                ]
            )

        async def call_tool(ctx, params):
            return CallToolResult(
                content=[TextContent(type="text", text="private-secret")],
                structured_content={"status": "recorded", "request_id": "draft-1:1", "filename": "private-secret"},
            )

        return Server("broken-test-server", on_list_tools=list_tools, on_call_tool=call_tool)

    async def test_sdk_structured_output_error_is_sanitized_even_inside_exception_group(self):
        client = MCPRecordingClient(self.config, client_factory=lambda: Client(self.broken_server()))
        result = await client.arecord("draft-1:1", "one")
        self.assertEqual(result.status, "unconfirmed")
        self.assertEqual(result.error_code, "invalid_response")
        self.assertEqual(result.attempts, 1)
        self.assertNotIn("private-secret", result.model_dump_json())

    def test_bad_startup_schema_has_a_safe_recording_error_not_an_exception_group(self):
        client = MCPRecordingClient(self.config, client_factory=lambda: Client(self.broken_server(bad_schema=True)))
        with self.assertRaises(RecordingError):
            client.check_ready()

    async def test_real_stalled_stdio_has_bounded_retries_and_reaps_both_processes(self):
        parameters = self.recorder.server_parameters.model_copy(update={"args": ["-c", "import time; time.sleep(60)"]})
        self.recorder.timeout = 0.2
        original = anyio.open_process
        original_open = anyio.open_file
        processes = []
        logs = []

        async def track(*args, **kwargs):
            process = await original(*args, **kwargs)
            processes.append(process)
            self.assertNotIn("AZURE_OPENAI_API_KEY", kwargs["env"])
            self.assertNotIn("LANGSMITH_API_KEY", kwargs["env"])
            return process

        async def track_log(*args, **kwargs):
            log = await original_open(*args, **kwargs)
            logs.append(log)
            return log

        started = time.monotonic()
        with (
            patch.object(MCPRecordingClient, "server_parameters", new_callable=PropertyMock, return_value=parameters),
            patch("mcp.client.stdio.anyio.open_process", side_effect=track),
            patch("recording_client.anyio.open_file", side_effect=track_log),
        ):
            result = await self.recorder.arecord("draft-1:1", "one")
        self.assertEqual(result.status, "unconfirmed")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.returncode is not None for process in processes))
        self.assertEqual(len(logs), 2)
        self.assertTrue(all(log.wrapped.closed for log in logs))
        self.assertLess(time.monotonic() - started, 15)

    async def test_real_stdio_cancellation_propagates_without_retry_or_leaking_process(self):
        parameters = self.recorder.server_parameters.model_copy(update={"args": ["-c", "import time; time.sleep(60)"]})
        original = anyio.open_process
        processes = []
        spawned = asyncio.Event()

        async def track(*args, **kwargs):
            process = await original(*args, **kwargs)
            processes.append(process)
            spawned.set()
            return process

        with (
            patch.object(MCPRecordingClient, "server_parameters", new_callable=PropertyMock, return_value=parameters),
            patch("mcp.client.stdio.anyio.open_process", side_effect=track),
        ):
            task = asyncio.create_task(self.recorder.arecord("draft-1:1", "one"))
            await asyncio.wait_for(spawned.wait(), 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)


if __name__ == "__main__":
    unittest.main()
