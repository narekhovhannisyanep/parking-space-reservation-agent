"""Request detection, location vocabulary, and "other options" follow-ups (offline)."""

import unittest

from langchain_core.messages import AIMessage, HumanMessage
from test_startup_and_suggestions import ParkingFixture, decode

from context_augmentation import ContextConfig, asks_availability, asks_other_options, public_retrieval_query


def query(text):
    return public_retrieval_query(text, 2000)


def is_request(text):
    return asks_availability([HumanMessage(text)], ContextConfig())


class RequestDetectionTests(unittest.TestCase):
    def test_common_request_wording_is_detected(self):
        for text in (
            "I need a parking spot.",
            "Where can I park?",
            "Any vacant places?",
            "Book a spot for me.",
            "Show me other options.",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_request(text))

    def test_information_only_wording_is_not_a_request(self):
        for text in (
            "Tell me about parking.",
            "What are the parking prices?",
            "What are the opening hours?",
            "Where is the elevator?",
            "Is parking free of charge?",
            "Hello!",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_request(text))


class LocationVocabularyTests(unittest.TestCase):
    def test_location_preferences_map_to_fixed_public_terms(self):
        for text, terms in (
            ("A spot near the stairwell", {"stairs"}),
            ("I want an outdoor spot", {"outdoor area"}),
            ("Something outside please", {"outdoor area"}),
            ("Reserve on the first level", {"first level"}),
            ("Any space on the 2nd floor?", {"second level"}),
            ("Along the north wall", {"north", "wall"}),
            ("By the west fence", {"west", "fence"}),
            ("In the corner near the walkway", {"corner", "pedestrian walkway"}),
            ("Close to the pedestrian lobby", {"lobby", "pedestrian walkway"}),
            ("Next to the gate", {"gate"}),
        ):
            with self.subTest(text=text):
                for term in terms:
                    self.assertIn(term, query(text))

    def test_query_contains_only_code_owned_terms(self):
        text = "Alice Smith ABC-123 wants a spot by the north stairs at secret-location"
        result = query(text)
        for private in ("Alice", "Smith", "ABC-123", "secret-location"):
            self.assertNotIn(private, result)
        self.assertEqual(query("Alice Smith ABC-123"), "parking information")

    def test_benefits_and_negated_stairs_have_public_queries(self):
        for text, expected in (
            ("I use a wheelchair", "step-free access limited mobility"),
            ("I want to avoid stairs", "step-free access limited mobility"),
            ("I carry shopping bags", "store shopping bags unloading"),
            ("I want to stay dry", "covered sheltered pedestrian route rain"),
            ("Away from busy traffic", "away from passing traffic perimeter walk"),
            ("I have a stroller", "pushchair young children"),
        ):
            with self.subTest(text=text):
                self.assertIn(expected, query(text))
        self.assertNotIn(" stairs ", " " + query("avoid stairs") + " ")

    def test_follow_up_detection_requires_other_or_more_options(self):
        self.assertTrue(asks_other_options([HumanMessage("Show me other options.")], ContextConfig()))
        self.assertTrue(asks_other_options([HumanMessage("Any different spots?")], ContextConfig()))
        self.assertTrue(asks_other_options([HumanMessage("More spaces, please")], ContextConfig()))
        self.assertFalse(asks_other_options([HumanMessage("Any available spaces?")], ContextConfig()))
        self.assertFalse(asks_other_options([HumanMessage("Other than prices, what about hours?")], ContextConfig()))


class FollowUpGraphTests(ParkingFixture):
    def setUp(self):
        super().setUp()
        self.agent.prepare_startup()

    def offered(self):
        return self.state()["offered_space_ids"]

    def test_other_options_exclude_all_previously_offered_spaces(self):
        self.turn("Any available spaces?")
        first = list(self.offered())
        self.turn("Show me other options.")
        second = list(self.offered())
        self.turn("Any other spaces?")
        third = list(self.offered())
        self.assertEqual([len(first), len(second), len(third)], [3, 3, 3])
        self.assertEqual(len(set(first + second + third)), 9)

    def test_fresh_request_starts_a_new_offer_round(self):
        self.turn("Any available spaces?")
        first = list(self.offered())
        self.turn("Show me other options.")
        self.turn("Any available spaces?")
        self.assertEqual(self.offered(), first)

    def test_exhausted_alternatives_offer_nothing_and_explain(self):
        ids = [identifier.split(":")[-1] for identifier in self.static.list_space_ids("demo-central")]
        self.dynamic.upsert_fact("demo-central", "availability", {identifier: False for identifier in ids[4:]})
        self.turn("Any available spaces?")
        self.turn("Other options?")
        self.assertEqual(len(self.offered()), 1)
        self.turn("Other options?")
        self.assertEqual(self.offered(), [])
        evidence = decode(self.state()["retrieval_context"])
        self.assertEqual(evidence["availability_options"], [])
        self.assertEqual(evidence["previous_offers_excluded"], 4)

    def test_follow_up_exclusion_is_thread_scoped_and_reset_by_cancel(self):
        self.turn("Any available spaces?", thread="alice")
        alice = list(self.offered())
        self.turn("Any available spaces?", thread="bob")
        self.assertEqual(self.state("bob")["offered_space_ids"], alice)
        self.turn("/cancel")
        self.turn("Show me other options.")
        self.assertEqual(self.offered(), alice)

    def test_other_options_without_previous_offers_is_an_ordinary_request(self):
        self.turn("Show me other options.")
        self.assertEqual(len(self.offered()), 3)
        self.assertNotIn("previous_offers_excluded", decode(self.state()["retrieval_context"]))

    def test_model_reply_is_not_used_to_derive_exclusions(self):
        self.model.invoke.side_effect = lambda *args, **kwargs: AIMessage("Try G-A01, G-A02 and OUT-V01.")
        self.turn("Any available spaces?")
        first = list(self.offered())
        self.turn("Show me other options.")
        self.assertFalse(set(first) & set(self.offered()))


if __name__ == "__main__":
    unittest.main()
