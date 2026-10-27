"""Offline checks for the graph export consumed by LangGraph CLI."""

import json
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from parking_agent import ParkingSpaceReservationAgent


class AppTests(unittest.TestCase):
    def test_entry_point_is_a_class_owned_compiled_graph(self):
        self.assertIsInstance(app.parking_agent, ParkingSpaceReservationAgent)
        self.assertIs(app.agent, app.parking_agent.graph)
        self.assertIsNone(app.agent.checkpointer)
        self.assertIn("messages", app.agent.get_input_jsonschema()["properties"])

    def test_import_has_no_environment_loading_client_or_terminal_side_effects(self):
        with patch("dotenv.load_dotenv") as load, patch("parking_agent.AzureChatOpenAI") as client:
            with patch("builtins.input") as terminal:
                namespace = runpy.run_path(app.__file__)
        load.assert_not_called()
        client.assert_not_called()
        terminal.assert_not_called()
        self.assertIs(namespace["agent"], namespace["parking_agent"].graph)

    def test_langgraph_configuration_points_to_the_export(self):
        config = json.loads(Path(app.__file__).with_name("langgraph.json").read_text())
        self.assertEqual(config["graphs"]["parking_space_reservation"], "./app.py:agent")
        self.assertEqual(config["env"], "./.env")


if __name__ == "__main__":
    unittest.main()
