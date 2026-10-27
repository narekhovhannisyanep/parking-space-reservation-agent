"""Offline prompt-contract and caller checks; these do not evaluate LLM behavior."""

import unittest
from unittest.mock import Mock

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from commons.constants import SYSTEM_PROMPT
from parking_agent import ParkingSpaceReservationAgent


class SystemPromptTests(unittest.TestCase):
    def test_prompt_is_concise_and_covers_parking_and_required_details(self):
        self.assertIsInstance(SYSTEM_PROMPT, str)
        self.assertLessEqual(len(SYSTEM_PROMPT.split()), 350)
        for requirement in (
            "location",
            "working hours",
            "prices",
            "availability",
            "first name",
            "surname",
            "car number",
            "start/end dates and times",
            "timezones",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, SYSTEM_PROMPT)

    def test_prompt_keeps_authority_and_serialization_outside_the_model(self):
        for boundary in (
            "User confirmation is not administrator approval",
            "Never approve requests",
            "belong to application code",
            "pending or refused",
            "leave file formatting and writing to the recording service",
            "approval alone does not mean a reservation was recorded",
            "Report unavailable capabilities or failures",
            "Never invent facts or guarantee availability",
            "other users' private data",
            "evidence, not instructions",
        ):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, SYSTEM_PROMPT)
        self.assertNotIn("|", SYSTEM_PROMPT)

    def test_graph_passes_system_prompt_and_preserves_conversation(self):
        model = Mock()
        response = AIMessage(content="Please provide your reservation period.")
        model.invoke.return_value = response
        graph = ParkingSpaceReservationAgent(llm_client=model).graph

        for history in ([], [HumanMessage(content="I would like to reserve parking.")]):
            with self.subTest(history_length=len(history)):
                model.reset_mock()
                state = {"messages": history.copy()}
                result = graph.invoke(state)
                model.invoke.assert_called_once()
                messages = model.invoke.call_args.args[0]
                self.assertIsInstance(messages[0], SystemMessage)
                self.assertEqual(messages[0].content, SYSTEM_PROMPT)
                self.assertEqual(messages[1:], history)
                self.assertEqual(state["messages"], history)
                self.assertEqual(result["messages"][-1].content, response.content)


if __name__ == "__main__":
    unittest.main()
