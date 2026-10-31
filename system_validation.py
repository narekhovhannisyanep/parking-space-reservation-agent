"""Stage 4 synthetic validation, NOT a new application entry point or public service.

Owns disposable stores/checkpoints, a localhost Agent Server process, and scripted
responses through the actual human-review interrupt. Offline runs inject stateless
models/embeddings but use real storage, graph, HTTP and stdio MCP. Live checks are
small and opt-in. Reports contain counts/timings/codes, never chats or private rows.
"""

import argparse
import asyncio
import hashlib
import ipaddress
import json
import math
import os
import platform
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from contextvars import ContextVar
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory, TemporaryFile, gettempdir
from threading import Barrier, Lock
from typing import Any, Self
from unittest.mock import patch

from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.errors import GraphInterrupt

from administrator_agent import ReviewWording
from commons.constants import DIAL_GPT_LUNA_6, DIAL_TEXT_EMBEDDING_3_SMALL
from context_augmentation import CONTEXT_HEADER
from parking_agent import ParkingSpaceReservationAgent
from rag_evaluation import CLAIM_PATTERN, UNKNOWN_PATTERN
from recording import RecordingConfig, RecordingInput, RecordingResult, recording_filename
from recording_client import MCPRecordingClient
from reservation_collection import DraftField, FieldUpdate, ReservationExtraction
from storage import DynamicParkingStore, StaticParkingStore, StorageConfig
from storage.reservations import ReservationRecord, ReservationStore

PROJECT = Path(__file__).resolve().parent
ASSISTANT = "parking_space_reservation"
MODEL_DELAY = 0.02
PHASES = (
    "reservation_collection",
    "context_augmentation",
    "parking_assitant",
    "administrator_agent",
    "administrator_review",
    "record_approved_reservation",
)
ACTIVE_PHASE: ContextVar[str] = ContextVar("stage4_active_phase", default="")
FALLBACK_WORDING = "Please approve or refuse the reservation request below."


class ValidationFailure(RuntimeError):
    """Fixed assertion/error code only; never attach provider or protocol contents."""

    def __init__(self, code: str, *, receipt: RecordingResult | None = None):
        if not re.fullmatch(r"[a-z0-9_]+", code):
            raise ValueError("Validation failures require a fixed safe code.")
        self.code = code
        self.recording_failure = (
            {
                "status": receipt.status,
                "error_code": receipt.error_code,
                "retryable": receipt.retryable,
                "attempts": receipt.attempts,
            }
            if receipt is not None
            else None
        )
        super().__init__(code)


def require(condition: object, code: str) -> None:
    if not condition:
        raise ValidationFailure(code)


def latency_summary(samples: Sequence[float]) -> dict:
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("Timing samples must be finite and nonnegative.")
    ordered = sorted(samples)
    if not ordered:
        return {"n": 0}
    return {
        "n": len(ordered),
        "median_ms": round(median(ordered) * 1000, 2),
        "p95_ms": round(ordered[math.ceil(len(ordered) * 0.95) - 1] * 1000, 2),
        "max_ms": round(ordered[-1] * 1000, 2),
    }


def maximum_concurrency(intervals: Sequence[tuple[float, float]]) -> int:
    events = []
    for start, end in intervals:
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Execution intervals must have finite ordered boundaries.")
        if end > start:
            events.extend(((start, 1), (end, -1)))
    current = maximum = 0
    for _, delta in sorted(events):  # Ends precede starts at identical boundaries.
        current += delta
        maximum = max(maximum, current)
    return maximum


def synthetic_extraction(messages: Sequence[BaseMessage]) -> ReservationExtraction:
    """A deliberately limited labeled grammar, not measured real-model accuracy."""
    text = next((message.text for message in reversed(messages) if isinstance(message, HumanMessage)), "")
    patterns: tuple[tuple[DraftField, str], ...] = (
        ("first_name", r"\bfirst name(?: to)?\s+([^;,.]+)"),
        ("surname", r"\bsurname(?: to)?\s+([^;,.]+)"),
        ("car_number", r"\bcar number(?: to)?\s+([^;,.]+)"),
        ("duration_hours", r"\b(?:for|duration:?(?: to)?)\s+([0-9]+(?:\.[0-9]+)?\s+hours?)\b"),
    )
    updates = [
        FieldUpdate(field=field, value=match[1].strip())
        for field, pattern in patterns
        if (match := re.search(pattern, text, re.IGNORECASE)) is not None
    ]
    return ReservationExtraction(
        intent="reservation"
        if updates or re.search(r"\b(reserve|reservation)\b", text, re.IGNORECASE)
        else "information",
        updates=updates,
    )


def synthetic_answer(messages: Sequence[BaseMessage]) -> AIMessage:
    # These are fixture sentinels, not an alternative privacy detector. Necessary
    # fields are allowed in extraction; they must not reach the public-answer model.
    for message in messages:
        if not isinstance(message, SystemMessage):
            require(not re.search(r"stage4-bike-\d+|\b(?:Demo|Tester|Example)\b", message.text), "private_model_input")
    text = next((message.text for message in reversed(messages) if isinstance(message, HumanMessage)), "")
    require(CONTEXT_HEADER in text, "missing_public_evidence")
    payload = json.loads(text.rsplit(CONTEXT_HEADER, 1)[1])
    parts = []
    for kind, label in (("working_hours", "Working hours"), ("prices", "Prices")):
        fact = payload.get("dynamic_facts", {}).get(kind, {})
        value = json.dumps(fact["value"], ensure_ascii=False) if fact.get("status") == "snapshot" else "unknown"
        parts.append(f"{label}: {value}.")
    if "availability_options" in payload:
        ids = [item["space_id"] for item in payload["availability_options"]]
        parts.append("Current parking options: " + (", ".join(ids) if ids else "none") + ". Suggestions only.")
    elif payload.get("static_spaces"):
        parts.append("Location evidence: " + " ".join(item["content"] for item in payload["static_spaces"]))
    return AIMessage(" ".join(parts))


class SyntheticEmbeddings(Embeddings):
    """Stateless hashed word vectors; real embedding accuracy is NOT benchmarked."""

    def embed_query(self, text: str) -> list[float]:
        require("stage4-bike-" not in text, "private_embedding_input")
        result = [0.0] * 64
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            result[hashlib.sha256(word.encode()).digest()[0] % 64] += 1
        return result if any(result) else [1.0, *([0.0] * 63)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class SyntheticChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "stage4-synthetic"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        time.sleep(MODEL_DELAY)
        return ChatResult(generations=[ChatGeneration(message=synthetic_answer(messages))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        await asyncio.sleep(MODEL_DELAY)
        return ChatResult(generations=[ChatGeneration(message=synthetic_answer(messages))])


def server_environment(root: Path, *, live: bool) -> dict[str, str]:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(root),
        "PYTHONPATH": str(PROJECT),
        "PYTHONUNBUFFERED": "1",
        "PARKING_CHROMA_DIRECTORY": str(root / "chroma"),
        "PARKING_SQLITE_PATH": str(root / "parking.sqlite3"),
        "PARKING_CHROMA_COLLECTION": "parking_spaces_v2",
        "PARKING_LOCATION_ID": "demo-central",
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGGRAPH_NO_VERSION_CHECK": "true",
        "DISABLE_TRUSTSTORE": "true",
    }
    if live:
        for key in (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "OPENAI_API_VERSION",
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            "AZURE_OPENAI_EMBEDDING_MODEL",
            "PARKING_EXTRACTION_METHOD",
        ):
            if value := os.environ.get(key):
                environment[key] = value
    return environment


def _workspace(root: Path) -> None:
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.parent != Path(gettempdir()).resolve()
        or not root.name.startswith("parking-stage4-")
    ):
        raise ValueError("Use a runner-owned parking-stage4 temporary workspace.")


class ValidationServer:
    """One server, never multiple startup-resetting processes sharing a ledger."""

    def __init__(self, root: Path, *, live: bool = False, jobs: int = 10):
        _workspace(root)
        if type(jobs) is not int or not 1 <= jobs <= 10:
            raise ValueError("Use 1-10 worker jobs for this bounded validation.")
        self.root = root
        self.live = live
        self.jobs = jobs
        self.storage = StorageConfig(chroma_directory=root / "chroma", sqlite_path=root / "parking.sqlite3")
        self.recording = RecordingConfig.from_storage(self.storage)
        self.process: subprocess.Popen | None = None
        self.diagnostics: Any = None
        self._diagnostics_context = ExitStack()
        self._group_cleaned = True
        self.port = 0
        self.startup_seconds = 0.0
        self.forced_cleanup = False

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def start(self) -> None:
        require(self.process is None or self.process.poll() is not None, "server_already_running")
        if self.live:
            require(os.getenv("PARKING_LIVE_TEST") == "1", "live_opt_in_required")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        ready_file = self.root / "ready.json"
        ready_file.unlink(missing_ok=True)
        with ExitStack() as stack:
            self.diagnostics = stack.enter_context(TemporaryFile(mode="w+b"))
            # Transfer ownership from startup to the full server lifespan.
            self._diagnostics_context = stack.pop_all()
        started = time.monotonic()
        ready = False
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--serve",
                    "--root",
                    str(self.root),
                    "--port",
                    str(self.port),
                    "--jobs",
                    str(self.jobs),
                    *(["--live-server"] if self.live else []),
                ],
                cwd=self.root,
                env=server_environment(self.root, live=self.live),
                stdout=self.diagnostics,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._group_cleaned = False
            deadline = started + (150 if self.live else 90)
            while time.monotonic() < deadline:
                require(self.process.poll() is None, "server_startup_failed")
                if ready_file.is_file():
                    try:
                        marker = json.loads(ready_file.read_text())
                        require(marker["pid"] == self.process.pid, "server_identity_mismatch")
                        self.port = marker["port"]
                        self.request("GET", "/ok", timeout=1)
                        ready = True
                        self.startup_seconds = time.monotonic() - started
                        return
                    except (json.JSONDecodeError, ValidationFailure):
                        pass  # Marker publication or socket binding may still be finishing.
                time.sleep(0.05)
            raise ValidationFailure("server_startup_timeout")
        finally:
            if not ready:
                self.stop()

    def stop(self) -> None:
        try:
            if self.process is not None and not self._group_cleaned:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=25)
                    except subprocess.TimeoutExpired:
                        self.forced_cleanup = True
                        os.killpg(self.process.pid, signal.SIGKILL)
                        self.process.wait(timeout=5)
                # Bound cleanup even if a failed server left children in its group.
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._group_cleaned = True
        finally:
            self._diagnostics_context.close()

    def request(self, method: str, path: str, body: dict | None = None, *, timeout: float | None = None) -> dict:
        require(path.startswith("/") and not path.startswith("//"), "invalid_local_path")
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        # Never send localhost requests through inherited corporate/public proxies.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=timeout or (180 if self.live else 45)) as response:
                contents = response.read(2_000_001)
        except urllib.error.HTTPError as error:
            error.close()
            raise ValidationFailure(f"http_{error.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ValidationFailure("http_transport_error") from None
        require(len(contents) <= 2_000_000, "oversized_http_response")
        try:
            value = json.loads(contents)
        except (ValueError, UnicodeError):
            raise ValidationFailure("invalid_http_json") from None
        require(isinstance(value, dict), "invalid_http_object")
        require("__error__" not in value, "graph_run_error")
        return value

    def events(self) -> list[dict]:
        path = self.root / "timings.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def fault(self, thread_id: str, mode: str = "") -> None:
        require(not self.live, "live_faults_forbidden")
        require(mode in {"", "before_write", "transport", "lost_receipt"}, "invalid_fixture_fault")
        path = self.root / "faults.json"
        values = json.loads(path.read_text()) if path.exists() else {}
        if mode:
            values[thread_id] = mode
        else:
            values.pop(thread_id, None)
        path.write_text(json.dumps(values))

    def resources(self) -> dict:
        require(self.process is not None and self.process.poll() is None, "server_not_running")
        assert self.process is not None
        process = Path("/proc") / str(self.process.pid)
        children = set()
        for path in process.glob("task/*/children"):
            children.update(path.read_text().split())
        return {"open_fds": len(list((process / "fd").iterdir())), "child_processes": len(children)}


def _instrument(agent: ParkingSpaceReservationAgent, root: Path) -> None:
    lock = Lock()

    def emit(phase: str, config: RunnableConfig, start: float, end: float, outcome: str) -> None:
        row = {
            "phase": phase,
            "thread_id": config.get("configurable", {}).get("thread_id", ""),
            "start": start,
            "end": end,
            "outcome": outcome,
        }
        with lock, (root / "timings.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(row) + "\n")

    def wrap(phase: str, sync: Callable, asynchronous: Callable) -> tuple[Callable, Callable]:
        def call(state, config: RunnableConfig | None = None):
            # The native async retrieval node calls its sync helper in to_thread.
            # Do not double-count that nested call or change its callable signature.
            if ACTIVE_PHASE.get() == phase:
                return sync(state) if phase == "context_augmentation" else sync(state, config)
            token = ACTIVE_PHASE.set(phase)
            started = time.monotonic()
            outcome = "error"
            try:
                value = sync(state) if phase == "context_augmentation" else sync(state, config)
                outcome = "completed"
                return value
            finally:
                ended = time.monotonic()
                if isinstance(sys.exception(), GraphInterrupt):
                    outcome = "interrupted"
                ACTIVE_PHASE.reset(token)
                emit(phase, config or {}, started, ended, outcome)

        async def acall(state, config: RunnableConfig):
            token = ACTIVE_PHASE.set(phase)
            started = time.monotonic()
            outcome = "error"
            try:
                value = await (asynchronous(state) if phase == "context_augmentation" else asynchronous(state, config))
                outcome = "completed"
                return value
            finally:
                ended = time.monotonic()
                if isinstance(sys.exception(), GraphInterrupt):
                    outcome = "interrupted"
                ACTIVE_PHASE.reset(token)
                await asyncio.to_thread(emit, phase, config, started, ended, outcome)

        call.__name__ = phase
        acall.__name__ = phase
        return call, acall

    for phase in PHASES:
        sync, asynchronous = wrap(phase, getattr(agent, "_" + phase), getattr(agent, "_a" + phase))
        setattr(agent, "_" + phase, sync)
        setattr(agent, "_a" + phase, asynchronous)
    agent.graph = agent._build_graph(None)


def _serve(root: Path, port: int, jobs: int, *, live: bool) -> None:
    """Child-only bootstrap: original app/graph/lifespan, fixture injection and timing."""
    from contextlib import asynccontextmanager

    _workspace(root)
    config = StorageConfig.from_environment()
    require(
        config.sqlite_path == root / "parking.sqlite3" and config.chroma_directory == root / "chroma",
        "unsafe_validation_storage",
    )
    require(1 <= port <= 65535 and 1 <= jobs <= 10, "invalid_validation_server_settings")
    with ExitStack() as stack:
        if not live:
            original_connect = socket.socket.connect
            original_connect_ex = socket.socket.connect_ex

            def local_connect(connection, address):
                if connection.family in {socket.AF_INET, socket.AF_INET6}:
                    require(ipaddress.ip_address(address[0]).is_loopback, "offline_outbound_network")
                return original_connect(connection, address)

            def local_connect_ex(connection, address):
                if connection.family in {socket.AF_INET, socket.AF_INET6}:
                    require(ipaddress.ip_address(address[0]).is_loopback, "offline_outbound_network")
                return original_connect_ex(connection, address)

            stack.enter_context(patch.object(socket.socket, "connect", local_connect))
            stack.enter_context(patch.object(socket.socket, "connect_ex", local_connect_ex))
        from langgraph_api.cli import run_server

        import app

        owner = app.parking_agent
        if not live:
            owner.static_store = StaticParkingStore(
                owner.config.storage, embeddings=SyntheticEmbeddings(), privacy_guard=owner.privacy_guard
            )
            owner._llm_client = SyntheticChatModel()

            async def extract(messages):
                await asyncio.sleep(MODEL_DELAY)
                return synthetic_extraction(messages)

            async def compose(_):
                await asyncio.sleep(MODEL_DELAY)
                return ReviewWording(message="Please review this reservation request.")

            owner._reservation_extractor = RunnableLambda(synthetic_extraction, afunc=extract)
            owner._administrator_composer = RunnableLambda(
                lambda _: ReviewWording(message="Please review this reservation request."), afunc=compose
            )

            class FaultClient(MCPRecordingClient):
                async def _call(self, value: RecordingInput) -> RecordingResult:
                    def mode():
                        path = root / "faults.json"
                        return json.loads(path.read_text()).get(value.thread_id, "") if path.exists() else ""

                    fault = await asyncio.to_thread(mode)
                    if fault == "before_write":
                        return RecordingResult(status="failed", request_id=value.request_id, error_code="storage_error")
                    if fault == "transport":
                        raise TimeoutError()
                    result = await super()._call(value)
                    if fault == "lost_receipt":
                        raise TimeoutError()  # Actual publication/replay, but neither receipt reaches the graph.
                    return result

            owner.recording_client = FaultClient(owner.config.recording)
        _instrument(owner, root)
        app.agent = owner.graph
        original_lifespan = app.lifespan

        @asynccontextmanager
        async def lifespan(application):
            async with original_lifespan(application):
                from urllib.parse import urlparse

                actual_port = urlparse(os.environ["LANGGRAPH_API_URL"]).port
                await asyncio.to_thread(
                    (root / "ready.json").write_text, json.dumps({"pid": os.getpid(), "port": actual_port})
                )
                yield

        app.webapp.router.lifespan_context = lifespan
        run_server(
            host="127.0.0.1",
            port=port,
            reload=False,
            graphs={ASSISTANT: "app:agent"},
            http={"app": "app:webapp"},
            n_jobs_per_worker=jobs,
            open_browser=False,
            server_level="ERROR",
            loop="asyncio",
            timeout_graceful_shutdown=15,
        )


class ScenarioRunner:
    def __init__(self, server: ValidationServer):
        self.server = server
        self.samples: dict[str, list[float]] = {}
        self.lock = Lock()
        self.sequence = 0
        self.ledger = ReservationStore(DynamicParkingStore(server.storage))

    def sample(self, phase: str, seconds: float) -> None:
        with self.lock:
            self.samples.setdefault(phase, []).append(seconds)

    def thread(self) -> tuple[str, str]:
        with self.lock:
            self.sequence += 1
            car = f"stage4-bike-{self.sequence}"
        return self.server.request("POST", "/threads", {})["thread_id"], car

    def state(self, thread: str) -> dict:
        return self.server.request("GET", f"/threads/{thread}/state")

    def run(self, thread: str, phase: str, **body: Any) -> dict:
        started = time.monotonic()
        try:
            return self.server.request(
                "POST",
                f"/threads/{thread}/runs/wait",
                {
                    "assistant_id": ASSISTANT,
                    "multitask_strategy": "enqueue",
                    "on_disconnect": "cancel",
                    **body,
                },
            )
        finally:
            self.sample(phase, time.monotonic() - started)

    def turn(self, thread: str, text: str) -> dict:
        phase = (
            "submission_to_review"
            if text.startswith("/submit ")
            else "manual_retry"
            if text == "/retry-recording"
            else "dialogue"
        )
        return self.run(thread, phase, input={"messages": [{"role": "user", "content": text}]})

    def interrupt(self, thread: str) -> dict:
        state = self.state(thread)
        require(len(state["tasks"]) == 1 and len(state["tasks"][0]["interrupts"]) == 1, "missing_review_interrupt")
        pending = state["tasks"][0]["interrupts"][0]
        schema = pending["response_schema"]
        require(schema["required"] == ["decision"] and set(schema["properties"]) == {"decision"}, "wrong_review_schema")
        require("default" not in schema["properties"]["decision"], "default_human_decision")
        return pending

    def resume(self, thread: str, value: dict) -> dict:
        pending = self.interrupt(thread)
        return self.run(thread, "resume_to_outcome", command={"resume": {pending["id"]: value}})

    def row(self, thread: str) -> ReservationRecord:
        require(self.server.storage.sqlite_path.is_file(), "missing_validation_ledger")
        request_id = self.state(thread)["values"]["submission"]["request_id"]
        row = self.ledger.get(request_id, thread)
        require(row is not None, "missing_canonical_request")
        assert row is not None
        return row

    def options(self, thread: str, *, other: bool = False) -> list[str]:
        self.turn(thread, "Show me other options." if other else "/options")
        values = self.state(thread)["values"]
        ids = values["offered_space_ids"]
        require(len(ids) <= 3 and len(ids) == len(set(ids)), "invalid_option_count")
        require(
            values["offer_duration_hours"] == values["reservation"]["request"]["duration_hours"],
            "unbound_offer_duration",
        )
        return ids

    def prepare(self, *, detailed: bool = False, duration: str = "2") -> str:
        thread, car = self.thread()
        self.turn(thread, "What are the working hours and prices?")
        values = self.state(thread)["values"]
        require(UNKNOWN_PATTERN.search(values["messages"][-1]["content"]), "missing_facts_not_unknown")
        require(not CLAIM_PATTERN.search(values["messages"][-1]["content"]), "premature_booking_claim")
        if detailed:
            self.turn(thread, "/submit G-A01")
            require(not self.state(thread)["values"].get("submission"), "incomplete_request_submitted")
        self.turn(
            thread,
            f"Reserve parking. First name Demo, surname Tester, car number {car}."
            + ("" if detailed else f" Reserve for {duration} hours."),
        )
        if detailed:
            before = self.state(thread)["values"]["reservation"]["draft"]
            self.turn(thread, "What are the prices?")
            values = self.state(thread)["values"]
            require(values["reservation"]["draft"] == before, "information_mutates_private_draft")
            require(
                car not in values["messages"][-1]["content"] and "Demo" not in values["messages"][-1]["content"],
                "private_answer_leak",
            )
            self.turn(thread, f"I want to reserve for {duration} hours.")
            previous = self.state(thread)["values"]["reservation"]["revision"]
            self.turn(thread, "Change surname to Example.")
            self.turn(thread, f"/confirm {previous}")
            require(self.state(thread)["values"]["reservation"]["status"] != "ready", "stale_revision_confirmed")
        values = self.state(thread)["values"]
        reservation = values["reservation"]
        require(reservation["status"] == "awaiting_confirmation", "details_not_collected")
        require(
            reservation["draft"]["car_number"] == car and reservation["draft"]["first_name"] == "Demo",
            "cross_thread_draft",
        )
        require(
            "start" not in reservation["request"] and "timezone" not in reservation["draft"],
            "scheduled_draft_reintroduced",
        )
        require(not values.get("submission"), "confirmation_is_submission")
        self.turn(thread, f"/confirm {reservation['revision']}")
        require(self.state(thread)["values"]["reservation"]["status"] == "ready", "confirmation_failed")
        return thread

    def submit(self, thread: str, space: str) -> dict:
        before = self.state(thread)["values"]["reservation"]["request"]
        self.turn(thread, f"/submit {space}")
        pending = self.interrupt(thread)
        row = self.row(thread)
        require(row.status == "pending" and row.space_id == space, "hold_not_pending_before_review")
        require(row.request.collected_details().model_dump(mode="json") == before, "hold_changes_collected_details")
        require(row.submitted_at == row.request.start, "wrong_submission_start")
        require(pending["value"]["request"] == row.request.model_dump(mode="json"), "noncanonical_review_facts")
        require(not (self.server.recording.directory / recording_filename(row.request_id)).exists(), "pending_recorded")
        return pending

    def outcome(self, thread: str, status: str) -> dict:
        state = self.state(thread)
        require(not state["tasks"], "unexpected_remaining_interrupt")
        row = self.row(thread)
        require(row.status == status, "wrong_final_status")
        reply = state["values"]["messages"][-1]["content"]
        receipt = state["values"]["recording"]
        path = self.server.recording.directory / recording_filename(row.request_id)
        if status == "approved":
            require(row.approval_time is not None and row.approval_time < row.request.end, "invalid_approval_instant")
            if receipt["status"] != "recorded":
                raise ValidationFailure("recording_not_confirmed", receipt=RecordingResult.model_validate(receipt))
            require("Recorded successfully" in reply, "missing_recorded_success_reply")
            self.check_file(row)
        else:
            require(not path.exists() and not receipt and "Recorded successfully" not in reply, "unapproved_recording")
        return {
            "status": status,
            "recorded": status == "approved",
            "recording_attempts": receipt.get("attempts", 0) if status == "approved" else 0,
        }

    def check_file(self, row: ReservationRecord) -> None:
        path = self.server.recording.directory / recording_filename(row.request_id)
        require(row.status == "approved" and row.approval_time is not None, "file_without_approval")
        assert row.approval_time is not None
        expected = (
            f"{row.request.first_name} {row.request.surname} | {row.request.car_number} | "
            f"{row.request.start.isoformat()} to {row.request.end.isoformat()} | {row.approval_time.isoformat()}\n"
        )
        require(
            path.is_file() and not path.is_symlink() and path.read_text(encoding="utf-8") == expected,
            "invalid_recorded_contents",
        )
        require(path.stat().st_mode & 0o777 == 0o600, "unsafe_recorded_permissions")

    def conversation(self, name: str, decision: str, *, detailed: bool = False) -> dict:
        require(decision in {"approve", "refuse"}, "invalid_scripted_decision")
        started = time.monotonic()
        thread = self.prepare(detailed=detailed)
        first = self.options(thread)
        require(first, "no_options_for_demo")
        if detailed:
            other = self.options(thread, other=True)
            require(other and not set(first).intersection(other), "repeated_other_options")
            first = other
        pending = self.submit(thread, first[0])
        original = self.row(thread)
        if detailed:
            self.resume(thread, {})
            require(self.row(thread) == original, "invalid_review_changes_ledger")
            require("validation_error" in self.interrupt(thread)["value"], "invalid_review_not_reprompted")
        wait = time.monotonic()
        time.sleep(0.05)
        self.sample("scripted_review_wait", time.monotonic() - wait)
        self.resume(thread, {"decision": decision})
        result = self.outcome(thread, "approved" if decision == "approve" else "refused")
        require(self.row(thread).request == original.request, "decision_shifts_reservation_period")
        self.sample("conversation_total", time.monotonic() - started)
        return {"case": name, **result, "review_wording_fallback": pending["value"]["message"] == FALLBACK_WORDING}

    def check_storage(self) -> dict:
        require(self.server.storage.sqlite_path.is_file(), "missing_validation_ledger")
        with closing(sqlite3.connect(self.server.storage.sqlite_path.as_uri() + "?mode=ro", uri=True)) as connection:
            overlaps = connection.execute(
                "SELECT COUNT(*) FROM parking_reservations a JOIN parking_reservations b "
                "ON a.request_id < b.request_id AND a.location_id = b.location_id AND a.space_id = b.space_id "
                "AND a.status IN ('pending','approved') AND b.status IN ('pending','approved') "
                "AND a.start_at < b.end_at AND b.start_at < a.end_at"
            ).fetchone()[0]
            approved = connection.execute(
                "SELECT request_id, thread_id FROM parking_reservations WHERE status='approved'"
            ).fetchall()
            public = connection.execute("SELECT value_json FROM parking_dynamic_facts").fetchall()
        require(overlaps == 0, "overlapping_allocations")
        require("stage4-bike-" not in json.dumps(public), "private_public_fact")
        files = list(self.server.recording.directory.iterdir())
        require(len(files) == len(approved), "wrong_approved_file_count")
        for request_id, thread in approved:
            row = self.ledger.get(request_id, thread)
            require(row is not None, "missing_approved_row")
            assert row is not None
            self.check_file(row)
        require(self.server.recording.directory.stat().st_mode & 0o777 == 0o700, "unsafe_output_directory")
        require(self.server.storage.sqlite_path.stat().st_mode & 0o777 == 0o600, "unsafe_ledger_permissions")
        resources = self.server.resources()
        require(resources["child_processes"] == 0, "unreaped_mcp_children")
        return {"approved": len(approved), "files": len(files), "overlaps": overlaps, "resources": resources}


def event_summary(events: Sequence[dict]) -> dict:
    return {
        "node_latency": {
            phase: latency_summary([event["end"] - event["start"] for event in events if event["phase"] == phase])
            for phase in PHASES
        },
        "maximum_node_concurrency": maximum_concurrency([(event["start"], event["end"]) for event in events]),
        "node_errors": sum(event["outcome"] == "error" for event in events),
    }


def integration_validation() -> dict:
    with TemporaryDirectory(prefix="parking-stage4-") as directory:
        with ValidationServer(Path(directory)) as server:
            runner = ScenarioRunner(server)
            cases = [
                runner.conversation("rag_multiturn_invalid_review_approve", "approve", detailed=True),
                runner.conversation("refusal", "refuse"),
            ]

            # Public CRUD and current-turn evidence, not an invented facility/pricing model.
            thread, _ = runner.thread()
            dynamic = DynamicParkingStore(server.storage)
            dynamic.upsert_fact("demo-central", "working_hours", "Monday to Friday 08:00-20:00")
            dynamic.upsert_fact("demo-central", "prices", {"hourly": 3, "currency": "EUR"})
            runner.turn(thread, "What are the working hours and prices?")
            reply = runner.state(thread)["values"]["messages"][-1]["content"]
            require(all(value in reply for value in ("08:00", "20:00", "3")), "public_fact_not_answered")
            dynamic.delete_fact("demo-central", "working_hours")
            dynamic.delete_fact("demo-central", "prices")
            cases.append({"case": "current_public_facts", "passed": True})

            # Offers bind to hours, not stale preview times, and changing hours invalidates them.
            thread = runner.prepare()
            stale = runner.options(thread)[0]
            runner.turn(thread, "Change duration to 3 hours.")
            session = runner.state(thread)["values"]["reservation"]
            runner.turn(thread, f"/confirm {session['revision']}")
            runner.turn(thread, f"/submit {stale}")
            require(not runner.state(thread)["values"].get("submission"), "stale_offer_submitted")
            chosen = runner.options(thread)[0]
            runner.submit(thread, chosen)
            runner.resume(thread, {"decision": "refuse"})
            runner.outcome(thread, "refused")
            cases.append({"case": "duration_invalidates_offer", "passed": True})

            for mode in ("before_write", "transport", "lost_receipt"):
                thread = runner.prepare()
                runner.submit(thread, runner.options(thread)[0])
                server.fault(thread, mode)
                runner.resume(thread, {"decision": "approve"})
                original = runner.row(thread)
                values = runner.state(thread)["values"]
                receipt = values["recording"]
                require(
                    original.status == "approved" and receipt["status"] != "recorded",
                    "failure_revokes_or_claims_recording",
                )
                require(receipt["attempts"] == (1 if mode == "before_write" else 2), "unbounded_recording_retries")
                require("/retry-recording" in values["messages"][-1]["content"], "missing_manual_retry_reply")
                path = server.recording.directory / recording_filename(original.request_id)
                require(path.exists() == (mode == "lost_receipt"), "wrong_uncertain_file_effect")
                runner.turn(thread, "What is the request status?")
                require(runner.state(thread)["values"]["recording"] == receipt, "ordinary_chat_retries_recording")
                server.fault(thread)
                runner.turn(thread, "/retry-recording")
                runner.outcome(thread, "approved")
                require(runner.row(thread) == original, "retry_changes_approval_or_period")
                recorded = runner.state(thread)["values"]["recording"]
                recording_calls = sum(event["phase"] == "record_approved_reservation" for event in server.events())
                runner.turn(thread, "/retry-recording")
                require(runner.state(thread)["values"]["recording"] == recorded, "recorded_receipt_retried")
                require(
                    sum(event["phase"] == "record_approved_reservation" for event in server.events())
                    == recording_calls,
                    "recorded_request_calls_mcp_again",
                )
                cases.append({"case": mode + "_manual_retry", "passed": True})

            previous = runner.row(thread)
            runner.turn(thread, "/new")
            values = runner.state(thread)["values"]
            require(not values["submission"] and not values["recording"], "new_reuses_previous_operation")
            runner.turn(
                thread,
                "Reserve parking. First name Demo, surname Tester, car number stage4-bike-800. Reserve for 2 hours.",
            )
            session = runner.state(thread)["values"]["reservation"]
            runner.turn(thread, f"/confirm {session['revision']}")
            runner.submit(thread, runner.options(thread)[0])
            require(runner.row(thread).request_id != previous.request_id, "sequential_request_reuses_identity")
            runner.resume(thread, {"decision": "refuse"})
            runner.outcome(thread, "refused")
            require(runner.ledger.get(previous.request_id, thread) == previous, "new_changes_previous_allocation")
            cases.append({"case": "sequential_new_request", "passed": True})

            thread = runner.prepare()
            chosen = runner.options(thread)[0]
            dynamic.upsert_fact("demo-central", "availability", False)
            runner.turn(thread, f"/submit {chosen}")
            values = runner.state(thread)["values"]
            require(not values.get("submission") and not values["offered_space_ids"], "public_override_not_rechecked")
            dynamic.delete_fact("demo-central", "availability")
            require(runner.options(thread), "default_availability_not_restored")
            cases.append({"case": "public_availability_recheck", "passed": True})

            thread = runner.prepare(duration="0.0003")
            runner.submit(thread, runner.options(thread)[0])
            end = runner.row(thread).request.end
            time.sleep(max(0, (end - datetime.now(UTC)).total_seconds()) + 0.03)
            runner.resume(thread, {"decision": "approve"})
            runner.outcome(thread, "elapsed")
            cases.append({"case": "elapsed_review", "passed": True})

            # Preserve actual file/checkpoint observations across a graceful server restart.
            approved_thread = runner.prepare()
            runner.submit(approved_thread, runner.options(approved_thread)[0])
            runner.resume(approved_thread, {"decision": "approve"})
            runner.outcome(approved_thread, "approved")
            pending_thread = runner.prepare()
            runner.submit(pending_thread, runner.options(pending_thread)[0])
            before = runner.check_storage()
            server.stop()
            server.start()
            runner.resume(pending_thread, {"decision": "approve"})
            runner.turn(approved_thread, "/retry-recording")
            for old in (approved_thread, pending_thread):
                require(
                    "startup reset" in runner.state(old)["values"]["messages"][-1]["content"],
                    "checkpoint_resurrects_request",
                )
            require(not list(server.recording.directory.iterdir()), "restart_resurrects_files")
            require(
                len(StaticParkingStore(server.storage).list_space_ids("demo-central")) == 20,
                "restart_changes_static_inventory",
            )
            cases.append({"case": "restart_old_pending_and_approved_checkpoints", "passed": True})
            result = {
                "cases": cases,
                "cases_passed": len(cases),
                "storage_before_restart": before,
                "latency": {key: latency_summary(value) for key, value in runner.samples.items()},
                **event_summary(server.events()),
            }
        require(not server.forced_cleanup, "forced_server_cleanup")
        return result


def load_validation(concurrency: int, sessions: int) -> dict:
    if not 1 <= concurrency <= 10 or not 1 <= sessions <= 30:
        raise ValueError("The bounded load profile supports 1-10 workers and 1-30 conversations.")
    with TemporaryDirectory(prefix="parking-stage4-") as directory:
        with ValidationServer(Path(directory), jobs=10) as server:
            runner = ScenarioRunner(server)
            warm, _ = runner.thread()
            runner.turn(warm, "What are the prices?")
            runner.samples = {}
            event_start = len(server.events())
            resources_before = server.resources()
            started = time.monotonic()
            results = []
            for offset in range(0, sessions, concurrency):
                count = min(concurrency, sessions - offset)
                barrier = Barrier(count)
                choices: set[str] = set()
                choices_lock = Lock()

                def worker(index, barrier=barrier, choices=choices, choices_lock=choices_lock):
                    finished = False
                    conversation_start = time.monotonic()
                    try:
                        thread = runner.prepare()
                        ids = runner.options(thread)
                        selected = None
                        for _ in range(7):  # At most the 20-space fixture's bounded offer rounds.
                            with choices_lock:
                                selected = next((space for space in ids if space not in choices), None)
                                if selected is not None:
                                    choices.add(selected)
                            if selected is not None:
                                break
                            ids = runner.options(thread, other=True)
                        require(selected is not None, "load_inventory_exhausted")
                        assert selected is not None
                        barrier.wait(timeout=45)
                        runner.submit(thread, selected)
                        wait_start = time.monotonic()
                        time.sleep(0.05)  # Labeled scripted review wait, not human throughput.
                        runner.sample("scripted_review_wait", time.monotonic() - wait_start)
                        decision = "approve" if index % 2 == 0 else "refuse"
                        runner.resume(thread, {"decision": decision})
                        outcome = runner.outcome(thread, "approved" if decision == "approve" else "refused")
                        runner.sample("conversation_total", time.monotonic() - conversation_start)
                        finished = True
                        return outcome
                    finally:
                        if not finished:
                            barrier.abort()

                with ThreadPoolExecutor(max_workers=count) as executor:
                    results.extend(executor.map(worker, range(offset, offset + count), timeout=120))
            elapsed = time.monotonic() - started
            storage = runner.check_storage()
            events = server.events()[event_start:]
            summary = event_summary(events)
            if concurrency > 1 and sessions > 1:
                require(summary["maximum_node_concurrency"] > 1, "load_was_serialized")
            require(summary["node_errors"] == 0, "load_node_errors")
            result = {
                "concurrency": concurrency,
                "conversations": sessions,
                "completed": len(results),
                "approved": sum(row["status"] == "approved" for row in results),
                "refused": sum(row["status"] == "refused" for row in results),
                "recording_attempts": sum(row["recording_attempts"] for row in results),
                "automatic_recording_retries": sum(row["recording_attempts"] for row in results)
                - sum(row["status"] == "approved" for row in results),
                "unexpected_errors": 0,
                "elapsed_seconds": round(elapsed, 3),
                "conversations_per_second": round(sessions / elapsed, 3),
                "startup_seconds_excluded": round(server.startup_seconds, 3),
                "resources_before": resources_before,
                "storage": storage,
                "latency": {key: latency_summary(value) for key, value in runner.samples.items()},
                **summary,
            }
        require(not server.forced_cleanup, "forced_server_cleanup")
        return result


def contention_validation(concurrency: int = 10) -> dict:
    with TemporaryDirectory(prefix="parking-stage4-") as directory:
        with ValidationServer(Path(directory)) as server:
            runner = ScenarioRunner(server)
            threads = [runner.prepare() for _ in range(concurrency)]
            offers = [runner.options(thread) for thread in threads]
            common = next(iter(set(offers[0]).intersection(*offers[1:])))
            barrier = Barrier(concurrency)

            def submit(thread):
                barrier.wait(timeout=45)
                runner.turn(thread, f"/submit {common}")
                return thread if runner.state(thread)["values"].get("submission") else None

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                winners = [thread for thread in executor.map(submit, threads, timeout=120) if thread]
            require(len(winners) == 1, "contention_multiple_holds")
            for loser in set(threads).difference(winners):
                values = runner.state(loser)["values"]
                require(
                    "overlapping" in values["messages"][-1]["content"] and common not in values["offered_space_ids"],
                    "contention_not_reported_safely",
                )
            require(runner.row(winners[0]).status == "pending", "contention_not_pending")
            require(not list(server.recording.directory.iterdir()), "contention_premature_recording")
            runner.resume(winners[0], {"decision": "refuse"})
            require(not runner.ledger.blocked_spaces("demo-central"), "refusal_does_not_release")
            storage = runner.check_storage()
            result = {
                "contenders": concurrency,
                "pending_winners": 1,
                "expected_conflicts": concurrency - 1,
                "released_after_refusal": True,
                "storage": storage,
            }
        require(not server.forced_cleanup, "forced_server_cleanup")
        return result


def live_validation() -> dict:
    require(os.getenv("PARKING_LIVE_TEST") == "1", "live_opt_in_required")
    with TemporaryDirectory(prefix="parking-stage4-") as directory:
        with ValidationServer(Path(directory), live=True, jobs=1) as server:
            runner = ScenarioRunner(server)
            cases = [
                runner.conversation("live_multiturn_approval", "approve", detailed=True),
                runner.conversation("live_refusal", "refuse"),
            ]
            require(not any(case["review_wording_fallback"] for case in cases), "live_administrator_fallback")
            result = {
                "cases": cases,
                "cases_passed": len(cases),
                "storage": runner.check_storage(),
                "startup_seconds": round(server.startup_seconds, 3),
                "latency": {key: latency_summary(value) for key, value in runner.samples.items()},
                **event_summary(server.events()),
                "live_provider_load_measured": False,
            }
        require(not server.forced_cleanup, "forced_server_cleanup")
        return result


def run_validation(part: str = "offline", *, sessions: int = 30, concurrency: Sequence[int] = (1, 5, 10)) -> dict:
    if part not in {"offline", "integration", "load", "live", "all"}:
        raise ValueError("Select a known validation part.")
    if (
        type(sessions) is not int
        or not 1 <= sessions <= 30
        or not concurrency
        or any(type(level) is not int or not 1 <= level <= 10 for level in concurrency)
    ):
        raise ValueError("Select a bounded nonempty validation profile.")
    report: dict = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "logical_cpus": os.cpu_count(),
            "packages": {
                name: version(name)
                for name in ("langgraph", "langgraph-api", "langgraph-cli", "mcp", "chromadb", "langchain-openai")
            },
        },
        "method": {
            "server": "localhost TCP Agent Server, one process, explicit worker job limit",
            "offline_models": "stateless labeled extraction, evidence answer and 64-dimensional hashed embeddings",
            "synthetic_model_delay_seconds": MODEL_DELAY,
            "review": "scripted decision-only interrupt resumes; Studio remains the selected human interface",
            "recording": "actual short-lived stdio MCP, including subprocess startup/cleanup",
            "load_model": "bounded closed-loop conversations; startup/warmup excluded, assertions/barriers included",
            "live_provider_load_measured": False,
        },
    }
    if part in {"all", "live"}:
        require(os.getenv("PARKING_LIVE_TEST") == "1", "live_opt_in_required")
        report["live_configuration"] = {
            "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT") or DIAL_GPT_LUNA_6,
            "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or DIAL_TEXT_EMBEDDING_3_SMALL,
            "extraction_method": os.getenv("PARKING_EXTRACTION_METHOD") or "function_calling",
            "provider_load_measured": False,
        }
    current_part = "integration"
    try:
        if part in {"offline", "all", "integration"}:
            report["integration"] = integration_validation()
        if part in {"offline", "all", "load"}:
            report["load"] = []
            for level in concurrency:
                current_part = f"load_{level}"
                report["load"].append(load_validation(level, sessions))
            current_part = "contention"
            report["contention"] = contention_validation()
        if part in {"all", "live"}:
            current_part = "live"
            report["live"] = live_validation()
    except ValidationFailure as error:
        report.update(passed=False, failed_part=current_part, error_code=error.code)
        if error.recording_failure is not None:
            report["recording_failure"] = error.recording_failure
    else:
        report["passed"] = True
    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4 isolated end-to-end/load validation; stdout JSON, no retained private data."
    )
    parser.add_argument("--part", choices=("offline", "integration", "load", "live", "all"), default="offline")
    parser.add_argument("--sessions", type=int, choices=range(1, 31), default=30)
    parser.add_argument("--concurrency", type=int, choices=(1, 5, 10), nargs="+", default=[1, 5, 10])
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--jobs", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--live-server", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.serve:
        _serve(args.root, args.port, args.jobs, live=args.live_server)
        return 0
    try:
        report = run_validation(args.part, sessions=args.sessions, concurrency=args.concurrency)
    except ValidationFailure as error:
        print(json.dumps({"passed": False, "error_code": error.code}))
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
