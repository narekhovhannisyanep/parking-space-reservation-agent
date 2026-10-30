"""Class-owned parking workflow resources; conversation state belongs to graph threads."""

import asyncio
import json
import os
import re
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
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from openai import APIError, OpenAIError
from pydantic import ValidationError

from administrator_agent import AdministratorAgent, AdministratorDecision, ReviewWording
from commons.constants import DIAL_GPT_LUNA_6, SYSTEM_PROMPT
from context_augmentation import (
    CONTEXT_HEADER,
    ContextConfig,
    asks_availability,
    asks_other_options,
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
    confirmation_summary,
    extraction_messages,
    is_collection_control,
    parse_extraction,
    validate_draft,
)
from storage import DynamicParkingStore, StaticParkingStore, StorageConfig
from storage.reservations import ReservationConflict, ReservationRecord, ReservationStore


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
    # All IDs offered since the latest fresh availability request, for "other options".
    offer_round_ids: list[str] = field(default_factory=list)
    privacy_values: list[str] = field(default_factory=list)
    offer_period: list[str] = field(default_factory=list)
    refresh_options: bool = False
    submission: dict = field(default_factory=dict)
    administrator_payload: dict = field(default_factory=dict)
    review_retry: bool = False


@dataclass(frozen=True)
class AgentConfig:
    """Shared, non-secret settings for one agent instance."""

    deployment: str = DIAL_GPT_LUNA_6
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
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT") or DIAL_GPT_LUNA_6,
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
        reservation_store: ReservationStore | None = None,
        administrator_composer: Runnable | None = None,
    ) -> None:
        self.config = config if config is not None else AgentConfig.from_environment()
        self._llm_client = llm_client
        self._reservation_extractor = reservation_extractor
        self._administrator_composer = administrator_composer
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
        self.reservation_store = (
            reservation_store
            if reservation_store is not None
            else ReservationStore(self.dynamic_store, clock=self._clock)
        )
        self.administrator_agent = AdministratorAgent(
            composer_factory=self._get_administrator_composer, privacy_guard=self.privacy_guard
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

    def _get_administrator_composer(self) -> Runnable:
        if self._administrator_composer is not None:
            return self._administrator_composer
        model = self._get_llm_client()
        with self._initialization_lock:
            if self._administrator_composer is None:
                try:
                    self._administrator_composer = model.with_structured_output(
                        ReviewWording, method=self.config.extraction_method, include_raw=True
                    )
                except NotImplementedError:
                    raise RuntimeError("The configured model does not support administrator communication.") from None
            return self._administrator_composer

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
            "refresh_options": False,
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
        if text.strip() in {"/cancel", "/new"} or any(
            getattr(previous.draft, field) != getattr(session.draft, field) for field in ("start", "end", "timezone")
        ):
            updates["offered_space_ids"] = []
            updates["offer_round_ids"] = []
            updates["offer_period"] = []
        return updates

    @staticmethod
    def _thread_id(config: RunnableConfig) -> str:
        value = config.get("configurable", {}).get("thread_id")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Reservation submission/review requires a persistent thread_id.")
        return value

    @staticmethod
    def _outcome_reply(row: ReservationRecord | None) -> str:
        if row is None:
            return "This request is unavailable after the startup reset. No approval can be applied. Use /new for a new request."
        prefix = f"Request {row.request_id}, space {row.space_id}: "
        if row.status == "pending":
            return (
                prefix
                + "pending human administrator review. The space is held for your period, not approved or recorded. Review in Studio Graph mode. Editing/cancellation is unavailable while waiting."
            )
        if row.status == "approved":
            return (
                prefix
                + f"approved by the human administrator at {row.approval_time.isoformat() if row.approval_time else ''}. The space remains allocated for your period. This reservation has not been recorded to a file; Stage 3 recording is not implemented. Use /new for a separate request."
            )
        if row.status == "refused":
            return (
                prefix
                + "refused by the human administrator. The hold has been released; nothing has been recorded. Use /new for a separate request."
            )
        return (
            prefix
            + "cannot be approved because its reservation start has passed. The hold has been released; nothing has been recorded. Use /new for a new period."
        )

    def _operational_input(self, state: ParkingState, text: str, config: RunnableConfig) -> dict | None:
        """Handle submission and existing operations before private field extraction."""
        command = text.strip()
        if state.submission:
            row = self.reservation_store.get(state.submission["request_id"], self._thread_id(config))
            if command.lower() == "/new" and (row is None or row.status != "pending"):
                updates = self._collection_result(state, "/new", ReservationExtraction())
                return {**updates, "submission": {}, "administrator_payload": {}}
            return {
                "reservation_reply": self._outcome_reply(row),
                "retrieval_context": "",
                "refresh_options": False,
                "submission": row.checkpoint() if row else {**state.submission, "status": "unavailable"},
            }
        if command.lower() == "/options":
            return {
                "reservation_reply": (
                    "Your details are confirmed. Select one currently offered space with /submit SPACE_ID."
                    if self._reservation_session(state).status == "ready"
                    else "Current suggestions only. Confirm your details, then select an offered space with /submit SPACE_ID."
                ),
                "retrieval_context": "",
                "refresh_options": True,
            }
        if not re.match(r"^/submit(?:\s|$)", command, flags=re.IGNORECASE):
            return None
        session = self._reservation_session(state)
        base = {"retrieval_context": "", "refresh_options": False}
        if session.status != "ready" or session.request is None:
            return {
                **base,
                "reservation_reply": "Collect and confirm your reservation details before submitting. User confirmation is not administrator approval.",
            }
        selected = re.fullmatch(r"/submit ([A-Za-z0-9][A-Za-z0-9_-]{0,63})", command, flags=re.IGNORECASE)
        if selected is None:
            return {**base, "reservation_reply": "Use /submit SPACE_ID with exactly one currently offered space ID."}
        current, issues = validate_draft(session, self.config.context.location_id, self._clock())
        if issues or current != session.request:
            return self._collection_result(state, "/confirm", ReservationExtraction())
        period = [session.request.start.isoformat(), session.request.end.isoformat()]
        if selected[1] not in state.offered_space_ids or period != state.offer_period:
            return {
                **base,
                "reservation_reply": "Choose a currently offered option for your confirmed period. Send /options, then /submit SPACE_ID.",
            }
        if not self.graph.checkpointer and not config.get("configurable", {}).get("__pregel_checkpointer"):
            # Agent Server supplies the runtime saver; standalone graphs must inject one.
            raise ValueError(
                "Reservation submission requires a checkpointer. Use Agent Server or inject a checkpointer."
            )
        try:
            ids = self.static_store.list_space_ids(session.request.location_id)
            row = self.reservation_store.hold(session.request, selected[1], self._thread_id(config), ids)
        except ReservationConflict as error:
            return {**base, "reservation_reply": str(error), "refresh_options": True, "offer_round_ids": []}
        return self._pending_updates(row)

    def _pending_updates(self, row: ReservationRecord) -> dict:
        return {
            "submission": row.checkpoint(),
            "reservation_reply": self._outcome_reply(row),
            "retrieval_context": "",
            "refresh_options": False,
            "administrator_payload": {},
        }

    def _reservation_collection(self, state: ParkingState, config: RunnableConfig) -> dict:
        self._check_startup()
        try:
            text = self._collection_input(state)
            operational = self._operational_input(state, text, config)
            if operational is not None:
                return operational
            safe_text = self.privacy_guard.redact(text, keep_names=True)
        except GuardrailError:
            return self._guardrail_failure(state)
        except (OSError, sqlite3.Error):
            raise RuntimeError(
                "Unable to read or submit the reservation. Check storage and retry; do not assume submission succeeded."
            ) from None
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
            operational = await asyncio.to_thread(self._operational_input, state, text, config)
            if operational is not None:
                return operational
            safe_text = await asyncio.to_thread(self.privacy_guard.redact, text, keep_names=True)
        except GuardrailError:
            return self._guardrail_failure(state)
        except (OSError, sqlite3.Error):
            raise RuntimeError(
                "Unable to read or submit the reservation. Check storage and retry; do not assume submission succeeded."
            ) from None
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
        if state.refresh_options:
            return "context_augmentation"
        if (
            state.reservation_reply
            and state.reservation_reply not in {EXTRACTION_FAILURE, SAFE_REPLY}
            and self._reservation_session(state).status in {"collecting", "ready"}
            and not state.submission
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
        if (
            session.status == "awaiting_confirmation"
            and session.request is not None
            and reply == confirmation_summary(session.request)
        ):
            updates["reservation"] = ReservationSession.model_validate(
                {**session.model_dump(), "summary_message_id": response.id}
            ).model_dump(mode="json")
        return updates

    def _context_augmentation(self, state: ParkingState) -> dict:
        # Exclusions come only from code-tracked offers, never from model replies.
        follow_up = asks_other_options(state.messages, self.config.context)
        previous = state.offer_round_ids if follow_up else []
        try:
            request = self._reservation_session(state).request
            period = [request.start.isoformat(), request.end.isoformat()] if request else []
            availability = state.refresh_options or asks_availability(state.messages, self.config.context)
            blocked = (
                self.reservation_store.blocked_spaces(
                    self.config.context.location_id,
                    request.start if request else None,
                    request.end if request else None,
                )
                if availability
                else []
            )
            context = retrieve_context(
                state.messages,
                self.static_store,
                self.dynamic_store,
                self.config.context,
                baseline_documents=self._baseline_documents,
                privacy_guard=self.privacy_guard,
                excluded_space_ids=previous,
                unavailable_space_ids=blocked,
                reservation_period=period,
                force_availability=state.refresh_options,
            )
        except GuardrailError:
            return self._guardrail_failure(state)
        except (OSError, sqlite3.Error):
            raise RuntimeError(
                "Unable to retrieve parking information. Check storage configuration or retry."
            ) from None
        updates: dict = {"retrieval_context": context, "refresh_options": False}
        options = offered_space_ids(context)
        if options is not None:
            updates["offered_space_ids"] = options
            updates["offer_round_ids"] = list(dict.fromkeys([*previous, *options]))
            updates["offer_period"] = period
        return updates

    def _route_response(self, state: ParkingState) -> Literal["administrator_agent", "administrator_review", "__end__"]:
        if state.submission.get("status") != "pending":
            return "__end__"
        return "administrator_review" if state.administrator_payload else "administrator_agent"

    def _review_record(self, state: ParkingState, config: RunnableConfig) -> ReservationRecord | None:
        self._check_startup()
        try:
            return self.reservation_store.get(state.submission["request_id"], self._thread_id(config))
        except (OSError, sqlite3.Error):
            raise RuntimeError(
                "Unable to load the administrator request. Check storage and retry; no decision has been applied."
            ) from None

    def _apply_decision(
        self, row: ReservationRecord, decision: AdministratorDecision, config: RunnableConfig
    ) -> ReservationRecord:
        try:
            return self.reservation_store.decide(row.request_id, self._thread_id(config), decision.decision)
        except (OSError, sqlite3.Error):
            raise RuntimeError(
                "Unable to finish the administrator decision. Check storage and retry; do not assume approval succeeded."
            ) from None

    def _administrator_agent(self, state: ParkingState, config: RunnableConfig) -> dict:
        row = self._review_record(state, config)
        if row is None or row.status != "pending":
            return {
                "administrator_payload": {},
                "submission": row.checkpoint() if row else {**state.submission, "status": "unavailable"},
                "reservation_reply": self._outcome_reply(row),
            }
        payload = self.administrator_agent.prepare(row.request, row.space_id, row.request_id, config)
        return {"administrator_payload": payload}

    async def _aadministrator_agent(self, state: ParkingState, config: RunnableConfig) -> dict:
        row = await asyncio.to_thread(self._review_record, state, config)
        if row is None or row.status != "pending":
            return {
                "administrator_payload": {},
                "submission": row.checkpoint() if row else {**state.submission, "status": "unavailable"},
                "reservation_reply": self._outcome_reply(row),
            }
        payload = await self.administrator_agent.aprepare(row.request, row.space_id, row.request_id, config)
        return {"administrator_payload": payload}

    def _administrator_review(self, state: ParkingState, config: RunnableConfig) -> dict:
        row = self._review_record(state, config)
        if row is not None and row.status == "pending":
            try:
                decision = interrupt(state.administrator_payload, response_schema=AdministratorDecision)
            except ValidationError:
                return self._invalid_review(state)
            if decision.request_id != row.request_id:
                return self._invalid_review(state)
            row = self._apply_decision(row, decision, config)
        return {
            "submission": row.checkpoint() if row else {**state.submission, "status": "unavailable"},
            "administrator_payload": {},
            "reservation_reply": self._outcome_reply(row),
            "retrieval_context": "",
            "review_retry": False,
        }

    async def _aadministrator_review(self, state: ParkingState, config: RunnableConfig) -> dict:
        row = await asyncio.to_thread(self._review_record, state, config)
        if row is not None and row.status == "pending":
            try:
                decision = interrupt(state.administrator_payload, response_schema=AdministratorDecision)
            except ValidationError:
                return self._invalid_review(state)
            if decision.request_id != row.request_id:
                return self._invalid_review(state)
            row = await asyncio.to_thread(self._apply_decision, row, decision, config)
        return {
            "submission": row.checkpoint() if row else {**state.submission, "status": "unavailable"},
            "administrator_payload": {},
            "reservation_reply": self._outcome_reply(row),
            "retrieval_context": "",
            "review_retry": False,
        }

    @staticmethod
    def _invalid_review(state: ParkingState) -> dict:
        return {
            "review_retry": True,
            "administrator_payload": {
                **state.administrator_payload,
                "validation_error": "Copy the displayed request_id and choose exactly approve or refuse. No decision was applied.",
            },
        }

    @staticmethod
    def _route_review(state: ParkingState) -> Literal["administrator_review", "parking_assitant"]:
        return "administrator_review" if state.review_retry else "parking_assitant"

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
        builder.add_node(
            "administrator_agent", RunnableLambda(self._administrator_agent, afunc=self._aadministrator_agent)
        )
        builder.add_node(
            "administrator_review", RunnableLambda(self._administrator_review, afunc=self._aadministrator_review)
        )
        builder.add_edge(START, "reservation_collection")
        builder.add_conditional_edges("reservation_collection", self._route_collection)
        builder.add_edge("context_augmentation", "parking_assitant")
        builder.add_conditional_edges("parking_assitant", self._route_response)
        builder.add_edge("administrator_agent", "administrator_review")
        builder.add_conditional_edges("administrator_review", self._route_review)
        # Agent Server supplies persistence. Injection is only needed for standalone use/tests.
        return builder.compile(checkpointer=checkpointer)
