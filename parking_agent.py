"""Class-owned parking workflow resources; conversation state belongs to graph threads."""

import asyncio
import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Annotated, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from openai import APIError, OpenAIError
from pydantic import ValidationError

from commons.constants import DIAL_GPT_LUNA_6, SYSTEM_PROMPT


@dataclass
class ParkingState:
    """Thread-scoped messages with LangGraph's append/update-by-ID semantics.

    A dataclass satisfies StateGraph's schema bound in ty without casting or
    suppressing the TypedDict protocol incompatibility of MessagesState.
    """

    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)


@dataclass(frozen=True)
class AgentConfig:
    """Shared, non-secret settings for one agent instance."""

    deployment: str = DIAL_GPT_LUNA_6
    system_prompt: str = SYSTEM_PROMPT
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    temperature: float = 0
    timeout: float = 60
    max_retries: int = 2

    @classmethod
    def from_environment(cls) -> "AgentConfig":
        """Read settings after LangGraph CLI has loaded the configured environment."""
        return cls(deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT") or DIAL_GPT_LUNA_6)


class ParkingSpaceReservationAgent:
    """Own shared config, model, and graph, never a shared user conversation.

    Future retrieval and reservation services will be initialized here and reused
    by graph nodes. No vector/SQL databases or approval services exist yet.
    """

    def __init__(
        self,
        llm_client: BaseChatModel | None = None,
        *,
        config: AgentConfig | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.config = config if config is not None else AgentConfig.from_environment()
        self._llm_client = llm_client
        self._initialization_lock = Lock()
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

    def _assistant(self, state: ParkingState, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
        model = self._get_llm_client()
        messages = [SystemMessage(self.config.system_prompt), *state.messages]
        try:
            response = model.invoke(messages, config=config)
        except APIError:
            # Avoid including raw provider errors in the graph's user-facing error.
            raise RuntimeError("Unable to get a response. Check the model configuration or retry the run.") from None
        return {"messages": [response]}

    async def _aassistant(self, state: ParkingState, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
        # Client construction can block; model requests use the native async API.
        model = await asyncio.to_thread(self._get_llm_client)
        messages = [SystemMessage(self.config.system_prompt), *state.messages]
        try:
            response = await model.ainvoke(messages, config=config)
        except APIError:
            raise RuntimeError("Unable to get a response. Check the model configuration or retry the run.") from None
        return {"messages": [response]}

    def _build_graph(self, checkpointer: BaseCheckpointSaver | None) -> CompiledStateGraph:
        builder = StateGraph(ParkingState)
        builder.add_node("assistant", RunnableLambda(self._assistant, afunc=self._aassistant))
        builder.add_edge(START, "assistant")
        builder.add_edge("assistant", END)
        # Agent Server supplies persistence. Injection is only needed for standalone use/tests.
        return builder.compile(checkpointer=checkpointer)
