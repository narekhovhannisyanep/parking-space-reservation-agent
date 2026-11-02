"""Minimal inventory validation and filtering need no databases or credentials."""

import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from typing import Any

from storage import ParkingSpace, SpaceFilters
from storage.seed import load_spaces


class ParkingSpaceTests(unittest.TestCase):
    def setUp(self):
        self.space = load_spaces()[0]

    def test_round_trip_and_canonical_identity(self):
        self.assertEqual(ParkingSpace.from_record(asdict(self.space)), self.space)
        doc = self.space.to_document()
        self.assertEqual(doc.id, "demo-central:space:G-A01")
        self.assertEqual(doc.metadata["space_id"], self.space.space_id)
        self.assertIn(self.space.description, doc.page_content)
        self.assertNotIn("description", doc.metadata)
        ParkingSpace.validate_document(doc)

    def test_minimal_fields_and_vehicle_agnostic_document(self):
        self.assertEqual(
            set(asdict(self.space)), {"space_id", "location_id", "label", "description", "source", "hourly_price_cents"}
        )
        doc = self.space.to_document()
        self.assertIn("Accepts any vehicle", doc.page_content)
        self.assertEqual(
            set(doc.metadata),
            {"space_id", "location_id", "label", "source", "record_type", "schema_version", "hourly_price_cents"},
        )
        self.assertEqual(doc.metadata["schema_version"], 3)

    def test_identity_is_scoped_by_location_and_immutable(self):
        other = replace(self.space, location_id="other")
        self.assertNotEqual(self.space.document_id, other.document_id)
        for name in asdict(self.space):
            with self.subTest(name=name), self.assertRaises(FrozenInstanceError):
                setattr(self.space, name, "changed")

    def test_invalid_fields_are_rejected_not_coerced(self):
        cases = {
            "space_id": ["", "x:y", " x", "x" * 65, None, 1],
            "location_id": [" ", "../other", 12],
            "label": [" ", None, True, "x" * 201],
            "description": [" ", "x" * 401, " untrimmed", []],
            "source": [None, " ", "x" * 201],
            "hourly_price_cents": [True, 0, 99, 2001, 500.0, "500", None],
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(field=name, value=value), self.assertRaises(ValueError):
                    replace(self.space, **{name: value})

    def test_private_obsolete_and_missing_fields_are_rejected(self):
        for name in (
            "car_number",
            "first_name",
            "status",
            "hold_id",
            "vehicle_type",
            "length_m",
            "width_m",
            "max_height_m",
            "accessible",
            "ev_charging",
            "covered",
            "level",
            "zone",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ParkingSpace.from_record({**asdict(self.space), name: "not needed"})
        for name in asdict(self.space):
            record = asdict(self.space)
            del record[name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                ParkingSpace.from_record(record)

    def test_invalid_metadata_and_mismatched_identity_are_rejected(self):
        for field, value in (
            ("schema_version", 1),
            ("schema_version", True),
            ("schema_version", 2.0),
            ("space_id", "other"),
            ("source", None),
            ("description", "wrong place"),
            ("occupied", False),
            ("length_m", 5),
            ("vehicle_type", "car"),
        ):
            doc = self.space.to_document()
            doc.metadata[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                ParkingSpace.validate_document(doc)


class SpaceFilterTests(unittest.TestCase):
    def test_base_filter_always_scopes_location_type_and_version(self):
        self.assertEqual(
            SpaceFilters().where("north"),
            {"$and": [{"location_id": "north"}, {"record_type": "parking_space"}, {"schema_version": 3}]},
        )

    def test_exact_space_filter_preserves_scope(self):
        self.assertEqual(
            SpaceFilters(space_id="A01").where("north")["$and"],
            [{"location_id": "north"}, {"record_type": "parking_space"}, {"schema_version": 3}, {"space_id": "A01"}],
        )

    def test_invalid_filter_values_and_removed_fields_are_rejected(self):
        values: list[Any] = [" ", "north:other", 1, True, [], "x" * 65]
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                SpaceFilters(space_id=value)
        locations: list[Any] = [" ", None, "north:other"]
        for location in locations:
            with self.subTest(location=location), self.assertRaises(ValueError):
                SpaceFilters().where(location)
        for name in (
            "vehicle_type",
            "vehicle_length_m",
            "vehicle_width_m",
            "vehicle_height_m",
            "accessible",
            "ev_charging",
            "covered",
            "level",
            "zone",
        ):
            with self.subTest(name=name), self.assertRaises(TypeError):
                invalid: Any = {name: "not needed"}
                SpaceFilters(**invalid)

    def test_filters_are_immutable_and_do_not_share_returned_queries(self):
        filters = SpaceFilters(space_id="A01")
        for name in asdict(filters):
            with self.subTest(name=name), self.assertRaises(FrozenInstanceError):
                setattr(filters, name, "changed")
        where = filters.where("north")
        where["$and"].clear()
        self.assertEqual(len(filters.where("north")["$and"]), 4)
