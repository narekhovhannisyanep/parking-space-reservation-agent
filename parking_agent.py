"""Class-owned parking workflow resources; conversation state belongs to graph threads."""

import asyncio
import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Annotated, Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from openai import APIError, OpenAIError
from pydantic import ValidationError

from commons.constants import DIAL_GPT_LUNA_6, SYSTEM_PROMPT
from context_augmentation import ContextConfig, augmented_messages, retrieve_context
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


@dataclass(frozen=True)
class AgentConfig:
    """Shared, non-secret settings for one agent instance."""

    deployment: str = DIAL_GPT_LUNA_6
    system_prompt: str = SYSTEM_PROMPT
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    temperature: float = 0
    timeout: float = 60
    max_retries: int = 2
    storage: StorageConfig = field(default_factory=StorageConfig)
    context: ContextConfig = field(default_factory=ContextConfig)

    @classmethod
    def from_environment(cls) -> "AgentConfig":
        """Read settings after LangGraph CLI has loaded the configured environment."""
        return cls(
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT") or DIAL_GPT_LUNA_6,
            storage=StorageConfig.from_environment(),
            context=ContextConfig(location_id=os.getenv("PARKING_LOCATION_ID") or "demo-central"),
        )


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
    ) -> None:
        self.config = config if config is not None else AgentConfig.from_environment()
        self._llm_client = llm_client
        self._initialization_lock = Lock()
        self.static_store = (
            static_store if static_store is not None else StaticParkingStore(self.config.storage, embeddings=embeddings)
        )
        self.dynamic_store = dynamic_store if dynamic_store is not None else DynamicParkingStore(self.config.storage)
        self.graph = self._build_graph(checkpointer)

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

    def _context_augmentation(self, state: ParkingState) -> dict[str, str]:
        return {
            "retrieval_context": retrieve_context(
                state.messages, self.static_store, self.dynamic_store, self.config.context
            )
        }

    async def _acontext_augmentation(self, state: ParkingState) -> dict[str, str]:
        # Chroma, embeddings, and SQLite are synchronous; keep them off the server event loop.
        return await asyncio.to_thread(self._context_augmentation, state)

    def _parking_assitant(self, state: ParkingState, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
        model = self._get_llm_client()
        messages = augmented_messages(self.config.system_prompt, state.messages, state.retrieval_context)
        try:
            response = model.invoke(messages, config=config)
        except APIError:
            # Avoid including raw provider errors in the graph's user-facing error.
            raise RuntimeError("Unable to get a response. Check the model configuration or retry the run.") from None
        return {"messages": [response]}

    async def _aparking_assitant(self, state: ParkingState, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
        # Client construction can block; model requests use the native async API.
        model = await asyncio.to_thread(self._get_llm_client)
        messages = augmented_messages(self.config.system_prompt, state.messages, state.retrieval_context)
        try:
            response = await model.ainvoke(messages, config=config)
        except APIError:
            raise RuntimeError("Unable to get a response. Check the model configuration or retry the run.") from None
        return {"messages": [response]}

    def _build_graph(self, checkpointer: BaseCheckpointSaver | None) -> CompiledStateGraph:
        builder = StateGraph(ParkingState, input_schema=ParkingMessages, output_schema=ParkingMessages)
        builder.add_node(
            "context_augmentation", RunnableLambda(self._context_augmentation, afunc=self._acontext_augmentation)
        )
        builder.add_node("parking_assitant", RunnableLambda(self._parking_assitant, afunc=self._aparking_assitant))
        builder.add_edge(START, "context_augmentation")
        builder.add_edge("context_augmentation", "parking_assitant")
        builder.add_edge("parking_assitant", END)
        # Agent Server supplies persistence. Injection is only needed for standalone use/tests.
        return builder.compile(checkpointer=checkpointer)
