"""Class-owned parking workflow resources; conversation state belongs to graph threads."""

import asyncio
import json
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Annotated, Literal
from uuid import uuid4

from chromadb.errors import ChromaError
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from openai import APIError, OpenAIError
from pydantic import ValidationError

from commons.constants import DIAL_GPT_SOL_6_1, SYSTEM_PROMPT
from context_augmentation import (
    CONTEXT_HEADER,
    ContextConfig,
    asks_availability,
    augmented_messages,
    latest_user_index,
    offered_space_ids,
    render_context,
    retrieve_context,
)
from guardrails import SAFE_REPLY, GuardrailError, PrivacyGuard, guarded_model_config
from reservation_collection import (
    EXTRACTION_FAILURE,
    ReservationExtraction,
    ReservationSession,
    advance_collection,
    extraction_messages,
    is_collection_control,
    parse_extraction,
)
from storage import DynamicParkingStore, StaticParkingStore, StorageConfig


@dataclass
class ParkingMessages:
    """Thread-scoped messages with LangGraph's append/update-by-ID semantics.

    A dataclass satisfies StateGraph's schema bound in ty without casting or
    suppressing the TypedDict protocol incompatibility of MessagesState.
    """

    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)


@dataclass
class ParkingState(ParkingMessages):
    """Internal evidence is replaced each turn, not appended to conversation history."""

    retrieval_context: str = ""
    reservation: dict = field(default_factory=dict)
    reservation_reply: str = ""
    offered_space_ids: list[str] = field(default_factory=list)
    privacy_values: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentConfig:
    """Shared, non-secret settings for one agent instance."""

    deployment: str = DIAL_GPT_SOL_6_1
    system_prompt: str = SYSTEM_PROMPT
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    temperature: float = 0
    timeout: float = 60
    max_retries: int = 2
    extraction_method: Literal["function_calling", "json_schema"] = "function_calling"
    storage: StorageConfig = field(default_factory=StorageConfig)
    context: ContextConfig = field(default_factory=ContextConfig)

    def __post_init__(self) -> None:
        if self.extraction_method not in {"function_calling", "json_schema"}:
            raise ValueError("PARKING_EXTRACTION_METHOD must be function_calling or json_schema.")

    @classmethod
    def from_environment(cls) -> "AgentConfig":
        """Read settings after LangGraph CLI has loaded the configured environment."""
        return cls(
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT") or DIAL_GPT_SOL_6_1,
            extraction_method=_extraction_method_from_environment(),
            storage=StorageConfig.from_environment(),
            context=ContextConfig(location_id=os.getenv("PARKING_LOCATION_ID") or "demo-central"),
        )


def _extraction_method_from_environment() -> Literal["function_calling", "json_schema"]:
    value = os.getenv("PARKING_EXTRACTION_METHOD") or "function_calling"
    if value not in {"function_calling", "json_schema"}:
        raise ValueError("PARKING_EXTRACTION_METHOD must be function_calling or json_schema.")
    return "json_schema" if value == "json_schema" else "function_calling"


class ParkingSpaceReservationAgent:
    """Own shared config, model, and graph, never a shared user conversation.

    Static/dynamic storage helpers are shared and lazy. Conversation state remains
    thread-scoped; retrieval is controlled by graph code, not model tools.
    """

    def __init__(
        self,
        llm_client: BaseChatModel | None = None,
        *,
        config: AgentConfig | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        embeddings: Embeddings | None = None,
        static_store: StaticParkingStore | None = None,
        dynamic_store: DynamicParkingStore | None = None,
        reservation_extractor: Runnable | None = None,
        clock: Callable[[], datetime] | None = None,
        require_startup: bool = False,
        privacy_guard: PrivacyGuard | None = None,
    ) -> None:
        self.config = config if config is not None else AgentConfig.from_environment()
        self._llm_client = llm_client
        self._reservation_extractor = reservation_extractor
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self._initialization_lock = Lock()
        self._startup_lock = Lock()
        self._startup_required = require_startup
        self.startup_ready = False
        self.baseline_context = ""
        self._baseline_documents: list[Document] | None = None
        self.privacy_guard = privacy_guard if privacy_guard is not None else PrivacyGuard()
        self.static_store = (
            static_store
            if static_store is not None
            else StaticParkingStore(self.config.storage, embeddings=embeddings, privacy_guard=self.privacy_guard)
        )
        self.dynamic_store = (
            dynamic_store
            if dynamic_store is not None
            else DynamicParkingStore(self.config.storage, privacy_guard=self.privacy_guard)
        )
        self.graph = self._build_graph(checkpointer)

    def prepare_startup(self) -> None:
        """Prepare shared stores/evidence once per server lifespan, before any chat."""
        with self._startup_lock:
            if self.startup_ready:
                return
            stage = "local privacy model preparation"
            try:
                self.privacy_guard.prepare()
                stage = "static database initialization/seeding"
                self.static_store.initialize()
                stage = "dynamic database reset"
                self.dynamic_store.reset()
                stage = "baseline context retrieval"
                ids = self.static_store.list_space_ids(self.config.context.location_id)
                if not ids:
                    raise ValueError("No inventory exists for the configured location.")
                documents = self.static_store.get_spaces(
                    ids[: min(3, self.config.context.retrieval_k)], location_id=self.config.context.location_id
                )
                facts = self.dynamic_store.list_facts(self.config.context.location_id)
                context = render_context(documents, facts, self.config.context, privacy_guard=self.privacy_guard)
                if not json.loads(context.removeprefix(CONTEXT_HEADER))["static_spaces"]:
                    raise ValueError("No baseline evidence fits the configured context budget.")
            except (OSError, ValueError, RuntimeError, sqlite3.Error, ChromaError, OpenAIError):
                # Startup is an all-or-fail boundary for storage/provider failures.
                # Do not expose raw provider exceptions, paths, or credentials.
                raise RuntimeError(
                    f"Parking startup failed during {stage}. Check storage paths, embedding configuration, "
                    "seed data, local privacy model installation, and PARKING_LOCATION_ID before restarting."
                ) from None
            self._baseline_documents = documents
            self.baseline_context = context
            self.startup_ready = True

    def finish_shutdown(self) -> None:
        """Invalidate readiness so another lifespan resets SQLite again, not Chroma."""
        with self._startup_lock:
            self.startup_ready = False
            self.baseline_context = ""
            self._baseline_documents = None

    def _check_startup(self) -> None:
        if self._startup_required and not self.startup_ready:
            raise RuntimeError("Parking databases are not ready. Complete server startup before accepting messages.")

    def _get_llm_client(self) -> BaseChatModel:
        # Import/graph construction stays offline; concurrent first runs share one client.
        with self._initialization_lock:
            if self._llm_client is None:
                try:
                    self._llm_client = AzureChatOpenAI(
                        azure_deployment=self.config.deployment,
                        reasoning_effort=self.config.reasoning_effort,
                        temperature=self.config.temperature,
                        timeout=self.config.timeout,
                        max_retries=self.config.max_retries,
                    )
                except (ValidationError, OpenAIError):
                    raise RuntimeError("Unable to initialize the model. Check the model configuration.") from None
            return self._llm_client

    def _get_reservation_extractor(self) -> Runnable:
        if self._reservation_extractor is not None:
            return self._reservation_extractor
        model = self._get_llm_client()
        with self._initialization_lock:
            if self._reservation_extractor is None:
                try:
                    self._reservation_extractor = model.with_structured_output(
                        ReservationExtraction, method=self.config.extraction_method, include_raw=True
                    )
                except NotImplementedError:
                    raise RuntimeError("The configured model does not support reservation extraction.") from None
            return self._reservation_extractor

    def _collection_input(self, state: ParkingState) -> str:
        index = latest_user_index(state.messages)
        return self.privacy_guard.text_content(state.messages[index]) if index is not None else ""

    def _private_values(self, state: ParkingState) -> list[str]:
        draft = self._reservation_session(state).draft
        return list(
            dict.fromkeys(
                [
                    *state.privacy_values,
                    *[value for value in (draft.first_name, draft.surname, draft.car_number) if value],
                ]
            )
        )

    def _guardrail_failure(self, state: ParkingState) -> dict:
        updates = self._collection_result(state, "", None)
        return {**updates, "reservation_reply": SAFE_REPLY, "offered_space_ids": []}

    def _reservation_session(self, state: ParkingState) -> ReservationSession:
        # Plain JSON checkpoint state avoids deserializing unregistered custom Python types.
        return ReservationSession.model_validate_json(json.dumps(state.reservation))

    def _collection_result(self, state: ParkingState, text: str, result: object) -> dict:
        previous = self._reservation_session(state)
        try:
            proposal = parse_extraction(result)
        except ValueError:
            session = ReservationSession.model_validate(
                {
                    **previous.model_dump(),
                    "status": "collecting" if previous.draft_id else "idle",
                    "request": None,
                    "confirmed_revision": None,
                    "summary_message_id": None,
                }
            )
            return {
                "reservation": session.model_dump(mode="json"),
                "reservation_reply": EXTRACTION_FAILURE,
                "retrieval_context": "",
            }
        session, reply = advance_collection(
            previous, text, proposal, self.config.context.location_id, self._clock(), state.messages
        )
        updates: dict = {
            "reservation": session.model_dump(mode="json"),
            "reservation_reply": reply,
            "retrieval_context": "",
            "privacy_values": list(
                dict.fromkeys(
                    [
                        *self._private_values(state),
                        *[
                            value
                            for value in (session.draft.first_name, session.draft.surname, session.draft.car_number)
                            if value
                        ],
                    ]
                )
            ),
        }
        if text.strip() in {"/cancel", "/new"}:
            updates["offered_space_ids"] = []
        return updates

    def _reservation_collection(self, state: ParkingState, config: RunnableConfig) -> dict:
        self._check_startup()
        try:
            text = self._collection_input(state)
            safe_text = self.privacy_guard.redact(text, keep_names=True)
        except GuardrailError:
            return self._guardrail_failure(state)
        if not text.strip():
            return {"reservation_reply": ""}
        if not safe_text.strip():
            return self._guardrail_failure(state)
        if is_collection_control(text):
            return self._collection_result(state, text, ReservationExtraction())
        extractor = self._get_reservation_extractor()
        try:
            result = extractor.invoke(
                extraction_messages(self._reservation_session(state), safe_text), config=guarded_model_config(config)
            )
        except (APIError, ValidationError, OutputParserException):
            return self._collection_result(state, text, None)
        return self._collection_result(state, safe_text, result)

    async def _areservation_collection(self, state: ParkingState, config: RunnableConfig) -> dict:
        self._check_startup()
        try:
            text = self._collection_input(state)
            safe_text = await asyncio.to_thread(self.privacy_guard.redact, text, keep_names=True)
        except GuardrailError:
            return self._guardrail_failure(state)
        if not text.strip():
            return {"reservation_reply": ""}
        if not safe_text.strip():
            return self._guardrail_failure(state)
        if is_collection_control(text):
            return self._collection_result(state, text, ReservationExtraction())
        extractor = await asyncio.to_thread(self._get_reservation_extractor)
        try:
            result = await extractor.ainvoke(
                extraction_messages(self._reservation_session(state), safe_text), config=guarded_model_config(config)
            )
        except (APIError, ValidationError, OutputParserException):
            return self._collection_result(state, text, None)
        return self._collection_result(state, safe_text, result)

    def _route_collection(self, state: ParkingState) -> Literal["context_augmentation", "parking_assitant"]:
        if (
            state.reservation_reply
            and state.reservation_reply not in {EXTRACTION_FAILURE, SAFE_REPLY}
            and self._reservation_session(state).status == "collecting"
            and asks_availability(state.messages, self.config.context)
        ):
            return "context_augmentation"
        return "parking_assitant" if state.reservation_reply else "context_augmentation"

    def _collection_response(self, state: ParkingState) -> dict:
        reply = state.reservation_reply
        options = offered_space_ids(state.retrieval_context)
        if options is not None:
            reply += "\n\n" + (
                "Current parking options: " + ", ".join(options) + ". These are suggestions, not holds or bookings."
                if options
                else "No eligible parking options can currently be offered."
            )
        response = AIMessage(reply, id=str(uuid4()))
        updates: dict = {"messages": [response]}
        session = self._reservation_session(state)
        if session.status == "awaiting_confirmation":
            updates["reservation"] = ReservationSession.model_validate(
                {**session.model_dump(), "summary_message_id": response.id}
            ).model_dump(mode="json")
        return updates

    def _context_augmentation(self, state: ParkingState) -> dict:
        try:
            context = retrieve_context(
                state.messages,
                self.static_store,
                self.dynamic_store,
                self.config.context,
                baseline_documents=self._baseline_documents,
                privacy_guard=self.privacy_guard,
            )
        except GuardrailError:
            return self._guardrail_failure(state)
        updates: dict = {"retrieval_context": context}
        options = offered_space_ids(context)
        if options is not None:
            updates["offered_space_ids"] = options
        return updates

    async def _acontext_augmentation(self, state: ParkingState) -> dict:
        # Chroma, embeddings, and SQLite are synchronous; keep them off the server event loop.
        return await asyncio.to_thread(self._context_augmentation, state)

    def _parking_assitant(self, state: ParkingState, config: RunnableConfig) -> dict:
        if state.reservation_reply:
            return self._collection_response(state)
        model = self._get_llm_client()
        try:
            history = self.privacy_guard.public_messages(state.messages, self._private_values(state))
            messages = augmented_messages(self.config.system_prompt, history, state.retrieval_context)
        except GuardrailError:
            return self._collection_response(ParkingState(reservation_reply=SAFE_REPLY))
        try:
            response = model.invoke(messages, config=guarded_model_config(config))
        except APIError:
            # Avoid including raw provider errors in the graph's user-facing error.
            raise RuntimeError("Unable to get a response. Check the model configuration or retry the run.") from None
        return {"messages": [self.privacy_guard.response(response, self._private_values(state))]}

    async def _aparking_assitant(self, state: ParkingState, config: RunnableConfig) -> dict:
        if state.reservation_reply:
            return self._collection_response(state)
        # Client construction can block; model requests use the native async API.
        model = await asyncio.to_thread(self._get_llm_client)
        try:
            history = await asyncio.to_thread(
                self.privacy_guard.public_messages, state.messages, self._private_values(state)
            )
            messages = augmented_messages(self.config.system_prompt, history, state.retrieval_context)
        except GuardrailError:
            return self._collection_response(ParkingState(reservation_reply=SAFE_REPLY))
        try:
            response = await model.ainvoke(messages, config=guarded_model_config(config))
        except APIError:
            raise RuntimeError("Unable to get a response. Check the model configuration or retry the run.") from None
        response = await asyncio.to_thread(self.privacy_guard.response, response, self._private_values(state))
        return {"messages": [response]}

    def _build_graph(self, checkpointer: BaseCheckpointSaver | None) -> CompiledStateGraph:
        builder = StateGraph(ParkingState, input_schema=ParkingMessages, output_schema=ParkingMessages)
        builder.add_node(
            "reservation_collection", RunnableLambda(self._reservation_collection, afunc=self._areservation_collection)
        )
        builder.add_node(
            "context_augmentation", RunnableLambda(self._context_augmentation, afunc=self._acontext_augmentation)
        )
        builder.add_node("parking_assitant", RunnableLambda(self._parking_assitant, afunc=self._aparking_assitant))
        builder.add_edge(START, "reservation_collection")
        builder.add_conditional_edges("reservation_collection", self._route_collection)
        builder.add_edge("context_augmentation", "parking_assitant")
        builder.add_edge("parking_assitant", END)
        # Agent Server supplies persistence. Injection is only needed for standalone use/tests.
        return builder.compile(checkpointer=checkpointer)
