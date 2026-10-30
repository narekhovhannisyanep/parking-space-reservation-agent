"""Class-resource and thread-isolation checks using mocked models only."""

import asyncio
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from threading import current_thread
from unittest.mock import AsyncMock, Mock, patch

from langchain.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeCancelledError
from openai import APIError, OpenAIError

from commons.constants import DIAL_GPT_SOL_6_1
from parking_agent import AgentConfig, ParkingSpaceReservationAgent, ParkingState
from reservation_collection import ReservationExtraction
from storage import StorageConfig


class ConfigTests(unittest.TestCase):
    def test_environment_deployment_and_default(self):
        for value in (None, "", "custom-deployment"):
            with self.subTest(value=value):
                env = {} if value is None else {"AZURE_OPENAI_DEPLOYMENT": value}
                with patch.dict(os.environ, env, clear=True):
                    self.assertEqual(AgentConfig.from_environment().deployment, value or DIAL_GPT_SOL_6_1)

    def test_shared_settings_are_immutable(self):
        config = AgentConfig()
        for field_name in (
            "deployment",
            "system_prompt",
            "reasoning_effort",
            "temperature",
            "timeout",
            "max_retries",
            "extraction_method",
            "storage",
            "context",
        ):
            with self.subTest(field=field_name), self.assertRaises(FrozenInstanceError):
                setattr(config, field_name, None)


class AgentTests(unittest.TestCase):
    def setUp(self):
        # Model lifecycle checks must not touch project data or embedding providers.
        extraction = patch.object(
            ParkingSpaceReservationAgent,
            "_get_reservation_extractor",
            return_value=RunnableLambda(lambda _: ReservationExtraction()),
        )
        extraction.start()
        self.addCleanup(extraction.stop)
        retrieval = patch("parking_agent.retrieve_context", return_value="")
        retrieval.start()
        self.addCleanup(retrieval.stop)

    def test_agent_owns_lazy_storage_without_initializing_on_construction(self):
        with patch("storage.static.Chroma") as chroma, patch("storage.dynamic.sqlite3.connect") as sqlite:
            model = Mock()
            model.invoke.return_value = AIMessage(content="Hello")
            config = AgentConfig(storage=StorageConfig(chroma_collection="custom_collection"))
            agent = ParkingSpaceReservationAgent(config=config, llm_client=model)
            self.assertIs(agent.static_store.config, config.storage)
            self.assertIs(agent.dynamic_store.config, config.storage)
            self.assertIsNone(agent.static_store._store)
            self.assertFalse(agent.dynamic_store._initialized)
            agent.graph.invoke({"messages": [HumanMessage(content="Hello")]})
            chroma.assert_not_called()
            sqlite.assert_not_called()

    def test_storage_and_embeddings_can_be_injected(self):
        static_store, dynamic_store, embeddings = Mock(), Mock(), Mock()
        agent = ParkingSpaceReservationAgent(static_store=static_store, dynamic_store=dynamic_store)
        self.assertIs(agent.static_store, static_store)
        self.assertIs(agent.dynamic_store, dynamic_store)
        agent = ParkingSpaceReservationAgent(embeddings=embeddings)
        self.assertIs(agent.static_store._embeddings, embeddings)

    def test_configuration_and_injected_client_are_reused(self):
        config = AgentConfig(system_prompt="Custom parking instructions")
        model = Mock()
        model.invoke.side_effect = [AIMessage(content="First"), AIMessage(content="Second")]
        with patch("parking_agent.AzureChatOpenAI") as constructor:
            agent = ParkingSpaceReservationAgent(llm_client=model, config=config)
            agent.graph.invoke({"messages": [HumanMessage(content="Hi")]})
            agent.graph.invoke({"messages": [HumanMessage(content="Again")]})
        constructor.assert_not_called()
        self.assertIs(agent.config, config)
        self.assertIs(agent._get_llm_client(), model)
        for call in model.invoke.call_args_list:
            self.assertEqual(call.args[0][0].content, config.system_prompt)

    def test_lazy_client_initialization_uses_instance_configuration(self):
        config = AgentConfig(deployment="custom", temperature=0.2, timeout=30, max_retries=1)
        with patch("parking_agent.AzureChatOpenAI") as constructor:
            agent = ParkingSpaceReservationAgent(config=config)
            constructor.assert_not_called()
            self.assertIs(agent._get_llm_client(), agent._get_llm_client())
        constructor.assert_called_once_with(
            azure_deployment="custom", reasoning_effort="medium", temperature=0.2, timeout=30, max_retries=1
        )

    def test_concurrent_initialization_creates_one_client(self):
        with patch("parking_agent.AzureChatOpenAI") as constructor:
            agent = ParkingSpaceReservationAgent()
            with ThreadPoolExecutor(max_workers=8) as executor:
                clients = list(executor.map(lambda _: agent._get_llm_client(), range(16)))
            constructor.assert_called_once()
            self.assertTrue(all(client is constructor.return_value for client in clients))

    def test_threads_preserve_history_without_sharing_it(self):
        model = Mock()
        model.invoke.side_effect = [
            AIMessage(content="Hello Alice"),
            AIMessage(content="Hello Bob"),
            AIMessage(content="Your name is Alice"),
        ]
        agent = ParkingSpaceReservationAgent(llm_client=model, checkpointer=InMemorySaver())
        alice: RunnableConfig = {"configurable": {"thread_id": "alice"}}
        bob: RunnableConfig = {"configurable": {"thread_id": "bob"}}
        agent.graph.invoke({"messages": [HumanMessage(content="I am Alice")]}, alice)
        agent.graph.invoke({"messages": [HumanMessage(content="I am Bob")]}, bob)
        agent.graph.invoke({"messages": [HumanMessage(content="What is my name?")]}, alice)
        last_prompt = model.invoke.call_args_list[-1].args[0]
        self.assertNotIn("Alice", str(last_prompt))
        self.assertNotIn("Bob", str(last_prompt))
        self.assertEqual(last_prompt[-1].content, "What is my name?")
        self.assertEqual(agent.graph.get_state(alice).values["messages"][0].content, "I am Alice")
        self.assertEqual(len(agent.graph.get_state(alice).values["messages"]), 4)
        self.assertEqual(len(agent.graph.get_state(bob).values["messages"]), 2)
        self.assertEqual(model.invoke.call_args.kwargs["config"]["configurable"]["thread_id"], "alice")

    def test_provider_error_is_sanitized_and_recovery_is_possible(self):
        model = Mock()
        model.invoke.side_effect = [
            APIError("private-api-key", request=Mock(), body=None),
            AIMessage(content="Recovered"),
        ]
        graph = ParkingSpaceReservationAgent(llm_client=model).graph
        with self.assertRaisesRegex(RuntimeError, "Unable to get a response") as error:
            graph.invoke({"messages": [HumanMessage(content="Hi")]})
        self.assertNotIn("private-api-key", str(error.exception))
        self.assertTrue(error.exception.__suppress_context__)
        self.assertEqual(
            graph.invoke({"messages": [HumanMessage(content="Retry")]})["messages"][-1].content, "Recovered"
        )

    def test_failed_initialization_can_be_retried(self):
        model = Mock()
        model.invoke.return_value = AIMessage(content="Recovered")
        with patch("parking_agent.AzureChatOpenAI", side_effect=[OpenAIError("private"), model]) as constructor:
            graph = ParkingSpaceReservationAgent().graph
            with self.assertRaisesRegex(RuntimeError, "Unable to initialize the model") as error:
                graph.invoke({"messages": [HumanMessage(content="Hi")]})
            self.assertNotIn("private", str(error.exception))
            graph.invoke({"messages": [HumanMessage(content="Retry")]})
        self.assertEqual(constructor.call_count, 2)

    def test_unexpected_model_errors_are_not_hidden(self):
        for failure in (TypeError("bug"), KeyError("bug"), ValueError("bug")):
            with self.subTest(failure=type(failure).__name__):
                model = Mock()
                model.invoke.side_effect = failure
                graph = ParkingSpaceReservationAgent(llm_client=model).graph
                with self.assertRaises(type(failure)) as error:
                    graph.invoke({"messages": [HumanMessage(content="Hi")]})
                self.assertIs(error.exception, failure)

    def test_missing_configuration_is_sanitized_without_loading_local_secrets(self):
        with patch.dict(os.environ, {}, clear=True):
            graph = ParkingSpaceReservationAgent().graph
            with self.assertRaisesRegex(RuntimeError, "Unable to initialize the model"):
                graph.invoke({"messages": [HumanMessage(content="Hi")]})

    def test_dataclass_defaults_are_not_shared_and_empty_input_is_supported(self):
        first, second = ParkingState(), ParkingState()
        first.messages.append(HumanMessage(content="Private to first state"))
        self.assertEqual(second.messages, [])
        model = Mock()
        model.invoke.return_value = AIMessage(content="Hello")
        result = ParkingSpaceReservationAgent(llm_client=model).graph.invoke({})
        self.assertEqual(result["messages"][-1].content, "Hello")
        self.assertEqual(len(model.invoke.call_args.args[0]), 1)

    def test_state_reducer_updates_messages_by_id(self):
        model = Mock()
        model.invoke.side_effect = [
            AIMessage(content="First answer", id="answer-1"),
            AIMessage(content="Second answer", id="answer-2"),
        ]
        graph = ParkingSpaceReservationAgent(llm_client=model, checkpointer=InMemorySaver()).graph
        config: RunnableConfig = {"configurable": {"thread_id": "message-update"}}
        graph.invoke({"messages": [HumanMessage(content="Original", id="question")]}, config)
        result = graph.invoke({"messages": [HumanMessage(content="Revised", id="question")]}, config)
        self.assertEqual(
            [message.content for message in result["messages"]], ["Revised", "First answer", "Second answer"]
        )


class AsyncAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        extraction = patch.object(
            ParkingSpaceReservationAgent,
            "_get_reservation_extractor",
            return_value=RunnableLambda(lambda _: ReservationExtraction()),
        )
        extraction.start()
        self.addCleanup(extraction.stop)
        retrieval = patch("parking_agent.retrieve_context", return_value="")
        retrieval.start()
        self.addCleanup(retrieval.stop)

    async def test_lazy_async_initialization_runs_off_the_event_loop_and_is_reused(self):
        loop_thread = current_thread()
        model = Mock()
        model.ainvoke = AsyncMock(side_effect=[AIMessage(content="One"), AIMessage(content="Two")])

        def initialize(**kwargs):
            self.assertIsNot(current_thread(), loop_thread)
            return model

        with patch("parking_agent.AzureChatOpenAI", side_effect=initialize) as constructor:
            graph = ParkingSpaceReservationAgent().graph
            await graph.ainvoke({"messages": [HumanMessage(content="Hi")]})
            await graph.ainvoke({"messages": [HumanMessage(content="Again")]})
        constructor.assert_called_once()
        self.assertEqual(model.ainvoke.await_count, 2)
        model.invoke.assert_not_called()

    async def test_async_cancellation_is_not_converted_to_a_provider_error(self):
        model = Mock()
        model.ainvoke = AsyncMock(side_effect=asyncio.CancelledError)
        graph = ParkingSpaceReservationAgent(llm_client=model).graph
        # LangGraph explicitly wraps a node-originated cancellation as a node error.
        with self.assertRaises(NodeCancelledError) as error:
            await graph.ainvoke({"messages": [HumanMessage(content="Hi")]})
        self.assertIsInstance(error.exception.__cause__, asyncio.CancelledError)

    async def test_cancelling_an_active_run_propagates_cancellation(self):
        started = asyncio.Event()

        async def blocked_model(*args, **kwargs):
            started.set()
            await asyncio.Future()

        model = Mock()
        model.ainvoke = AsyncMock(side_effect=blocked_model)
        graph = ParkingSpaceReservationAgent(llm_client=model).graph
        task = asyncio.create_task(graph.ainvoke({"messages": [HumanMessage(content="Hi")]}))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
        finally:
            task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_async_runs_use_native_model_api_and_isolate_threads(self):
        model = Mock()
        model.ainvoke = AsyncMock(side_effect=[AIMessage(content="One"), AIMessage(content="Two")])
        agent = ParkingSpaceReservationAgent(llm_client=model, checkpointer=InMemorySaver())
        configs: list[RunnableConfig] = [{"configurable": {"thread_id": name}} for name in ("one", "two")]
        results = await asyncio.gather(
            *[
                agent.graph.ainvoke({"messages": [HumanMessage(content=name)]}, config)
                for name, config in zip(("One", "Two"), configs)
            ]
        )
        model.invoke.assert_not_called()
        self.assertEqual(model.ainvoke.await_count, 2)
        for call in model.ainvoke.call_args_list:
            self.assertEqual(len(call.args[0]), 2)
        self.assertTrue(all(len(result["messages"]) == 2 for result in results))

    async def test_async_error_is_sanitized(self):
        model = Mock()
        model.ainvoke = AsyncMock(side_effect=APIError("private-api-key", request=Mock(), body=None))
        graph = ParkingSpaceReservationAgent(llm_client=model).graph
        with self.assertRaisesRegex(RuntimeError, "Unable to get a response") as error:
            await graph.ainvoke({"messages": [HumanMessage(content="Hi")]})
        self.assertNotIn("private-api-key", str(error.exception))

    async def test_unexpected_async_model_errors_are_not_hidden(self):
        for failure in (TypeError("bug"), KeyError("bug"), ValueError("bug")):
            with self.subTest(failure=type(failure).__name__):
                model = Mock()
                model.ainvoke = AsyncMock(side_effect=failure)
                graph = ParkingSpaceReservationAgent(llm_client=model).graph
                with self.assertRaises(type(failure)) as error:
                    await graph.ainvoke({"messages": [HumanMessage(content="Hi")]})
                self.assertIs(error.exception, failure)


if __name__ == "__main__":
    unittest.main()
