"""Private stdio MCP server: one approved-recording tool, no arbitrary filesystem API."""

import argparse
import asyncio
import sys
from pathlib import Path

from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams, TextContent, Tool
from pydantic import ValidationError

from recording import ApprovedReservationWriter, RecordingConfig, RecordingError, RecordingInput, RecordingResult

TOOL_NAME = "record_approved_reservation"


def create_server(config: RecordingConfig) -> Server:
    """Explicit startup only. Imports do not create storage or launch a server."""
    writer = ApprovedReservationWriter(config)
    writer.check_ready()
    tool = Tool(
        name=TOOL_NAME,
        description="Record an existing human-approved reservation from the local private ledger.",
        input_schema=RecordingInput.model_json_schema(),
        output_schema=RecordingResult.model_json_schema(),
    )

    async def list_tools(ctx: ServerRequestContext, params: PaginatedRequestParams | None) -> ListToolsResult:
        return ListToolsResult(tools=[tool])

    async def call_tool(ctx: ServerRequestContext, params: CallToolRequestParams) -> CallToolResult:
        # Low-level SDK is intentional: exact extra-forbidden input and structured
        # error codes plus is_error. Validate here; advertised schemas alone do not.
        try:
            if params.name != TOOL_NAME:
                raise ValueError("Unsupported tool.")
            value = RecordingInput.model_validate(params.arguments or {})
        except (ValueError, ValidationError):
            result = RecordingError("invalid_request").result("")
        else:
            try:
                result = await asyncio.to_thread(writer.record, value.request_id, value.thread_id)
            except RecordingError as error:
                result = error.result(value.request_id)
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="Recording confirmed." if result.status == "recorded" else "Recording not confirmed.",
                )
            ],
            structured_content=result.model_dump(mode="json"),
            is_error=result.status != "recorded",
        )

    return Server("parking-approved-recording", version="1", on_list_tools=list_tools, on_call_tool=call_tool)


async def serve(config: RecordingConfig) -> None:
    server = create_server(config)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(description="Private approved-reservation MCP server (stdio only).")
    parser.add_argument("--sqlite-path", type=Path, required=True)
    parser.add_argument("--chroma-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = RecordingConfig(args.output_directory, args.sqlite_path, args.chroma_directory)
        asyncio.run(serve(config))
    except (OSError, ValueError, RuntimeError):
        # stdout is exclusively MCP. Never print raw path/SQL/validation errors.
        print("Recording server failed. Check its local storage configuration and permissions.", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
