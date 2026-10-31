"""Bounded code-selected MCP calls. No tools or record contents are handed to an LLM."""

import asyncio
import os
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TextIO, cast

import anyio
from mcp import Client, MCPError, StdioServerParameters, stdio_client
from mcp.types import CallToolResult
from pydantic import ValidationError

from recording import RecordingConfig, RecordingError, RecordingErrorCode, RecordingInput, RecordingResult
from recording_server import TOOL_NAME

TRANSPORT_ERRORS = (
    TimeoutError,
    OSError,
    MCPError,
    anyio.EndOfStream,
    anyio.BrokenResourceError,
    anyio.ClosedResourceError,
)


class MCPRecordingClient:
    """Short-lived native async sessions, owned/closed per attempt, including startup."""

    def __init__(
        self,
        config: RecordingConfig,
        *,
        timeout: float = 15,
        client_factory: Callable[[], Client] | None = None,
    ) -> None:
        if not 0 < timeout <= 60:
            raise ValueError("Recording timeout must be positive and no greater than 60 seconds.")
        self.config = config
        self.timeout = timeout
        self.client_factory = client_factory

    @property
    def server_parameters(self) -> StdioServerParameters:
        project = Path(__file__).resolve().parent
        return StdioServerParameters(
            command=sys.executable,
            args=[
                str(project / "recording_server.py"),
                "--sqlite-path",
                str(self.config.sqlite_path),
                "--chroma-directory",
                str(self.config.chroma_directory),
                "--output-directory",
                str(self.config.directory),
            ],
            cwd=str(project),
            # SDK inherits only an OS allow-list, not os.environ. Never pass API
            # keys or load .env in the child. Keep provider tracing disabled too.
            env={"PYTHONUNBUFFERED": "1", "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"},
        )

    @asynccontextmanager
    async def _connection(self) -> AsyncGenerator[Client]:
        if self.client_factory is not None:
            async with self.client_factory() as client:
                yield client
        else:
            # Do not relay unexpected SDK/child exception text into user replies
            # or Agent Server logs. Startup errors identify the failed stage.
            errlog = await anyio.open_file(os.devnull, "w")
            try:
                # AnyIO types this as IO[str]; mode='w' gives a TextIO object,
                # which is the narrower annotation used by the SDK transport.
                async with Client(
                    stdio_client(self.server_parameters, errlog=cast(TextIO, errlog.wrapped)),
                    read_timeout_seconds=self.timeout,
                ) as client:
                    yield client
            finally:
                # A deadline may already have cancelled the enclosing scope.
                # Closing the wrapper must complete even in that case.
                with anyio.CancelScope(shield=True):
                    await errlog.aclose()

    async def _acheck_ready(self) -> None:
        # SDK cleanup uses AnyIO shielding. A native asyncio deadline can
        # interrupt that shield during shutdown; use its cancellation model.
        with anyio.fail_after(self.timeout):
            async with self._connection() as client:
                tools = (await client.list_tools()).tools
                if (
                    len(tools) != 1
                    or tools[0].name != TOOL_NAME
                    or tools[0].input_schema != RecordingInput.model_json_schema()
                    or tools[0].output_schema != RecordingResult.model_json_schema()
                ):
                    raise RecordingError("invalid_response")

    def check_ready(self) -> None:
        try:
            asyncio.run(self._acheck_ready())
        except (ExceptionGroup, RuntimeError, *TRANSPORT_ERRORS) as error:
            failure = self._exception_result(error, "")
            if failure is None:
                raise
            assert failure.error_code is not None
            raise RecordingError(failure.error_code) from None

    @staticmethod
    def _unconfirmed(
        request_id: str, code: RecordingErrorCode = "invalid_response", *, retryable: bool = False
    ) -> RecordingResult:
        return RecordingResult(status="unconfirmed", request_id=request_id, error_code=code, retryable=retryable)

    @classmethod
    def _exception_result(cls, error: BaseException, request_id: str) -> RecordingResult | None:
        """Recognize safe boundary failures, including SDK task-group wrappers.

        Unknown programming errors and cancellation are never swallowed. A
        known schema error may quote private returned data, so keep only a code.
        """
        if isinstance(error, BaseExceptionGroup):
            failures = []
            for child in error.exceptions:
                failure = cls._exception_result(child, request_id)
                if failure is None:
                    return None
                failures.append(failure)
            return next((failure for failure in failures if not failure.retryable), failures[0])
        if isinstance(error, RecordingError):
            return error.result(request_id)
        if isinstance(error, TRANSPORT_ERRORS):
            return cls._unconfirmed(request_id, "transport_error", retryable=True)
        if isinstance(error, RuntimeError) and str(error).startswith("Invalid structured content returned by tool"):
            return cls._unconfirmed(request_id)
        return None

    @classmethod
    def validate_result(cls, result: object, request_id: str) -> RecordingResult:
        if not isinstance(result, CallToolResult):
            return cls._unconfirmed(request_id)
        try:
            receipt = RecordingResult.model_validate(result.structured_content)
        except ValidationError:
            return cls._unconfirmed(request_id)
        if receipt.request_id != request_id or result.is_error != (receipt.status != "recorded"):
            return cls._unconfirmed(request_id)
        # Neither arbitrary content blocks nor protocol metadata are retained.
        return receipt

    async def _call(self, value: RecordingInput) -> RecordingResult:
        with anyio.fail_after(self.timeout):
            async with self._connection() as client:
                result = await client.call_tool(TOOL_NAME, value.model_dump())
                return self.validate_result(result, value.request_id)

    async def arecord(self, request_id: str, thread_id: str) -> RecordingResult:
        try:
            value = RecordingInput(request_id=request_id, thread_id=thread_id)
        except ValidationError:
            return RecordingError("invalid_request").result("")
        for attempt in (1, 2):
            try:
                receipt = await self._call(value)
            except (ExceptionGroup, RuntimeError, *TRANSPORT_ERRORS) as error:
                failure = self._exception_result(error, value.request_id)
                if failure is None:
                    raise
                receipt = failure
            if receipt.request_id != value.request_id:
                receipt = self._unconfirmed(value.request_id)
            receipt = RecordingResult.model_validate({**receipt.model_dump(), "attempts": attempt})
            if receipt.status == "recorded" or not receipt.retryable or attempt == 2:
                return receipt
        raise AssertionError("Unreachable recording retry loop.")

    def record(self, request_id: str, thread_id: str) -> RecordingResult:
        return asyncio.run(self.arecord(request_id, thread_id))
