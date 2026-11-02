"""Offline artifact checks; run with the isolated presentation tools, not the app."""

import hashlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from build_presentation import ASSETS, Canvas, build_deck, validate_package
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.before = {name: hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() for name in ASSETS.glob("*.png")}
        cls.report = build_deck(cls.output)
        cls.deck = Presentation(cls.output / "Parking_Space_Reservation_Agent.pptx")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_fourteen_widescreen_slides_and_author(self):
        self.assertEqual(len(self.deck.slides), 14)
        self.assertAlmostEqual(self.deck.slide_width / self.deck.slide_height, 16 / 9)
        self.assertEqual(self.deck.core_properties.author, "Narek Hovhannisyan")
        titles = [slide.shapes.title.text for slide in self.deck.slides]
        self.assertEqual(len(set(titles)), 14)

    def test_ten_minute_notes_are_embedded_and_exported(self):
        self.assertEqual(self.report["planned_seconds"], 600)
        notes_file = (self.output / "SPEAKER_NOTES.md").read_text()
        for slide in self.deck.slides:
            self.assertTrue(slide.has_notes_slide)
            notes = slide.notes_slide.notes_text_frame.text
            self.assertIn("Suggested time:", notes)
            self.assertIn("Sources:", notes)
            self.assertIn(slide.shapes.title.text.replace("\n", " "), notes_file)
        self.assertGreater(len(notes_file.split()), 1000)

    def test_all_seven_approved_assets_used_without_modification(self):
        self.assertEqual(len(self.report["source_sha256"]), 7)
        for name, digest in self.report["source_sha256"].items():
            path = ASSETS / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertEqual(self.before[path], digest)
        pictures = [
            shape for slide in self.deck.slides for shape in slide.shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE
        ]
        self.assertGreaterEqual(len(pictures), 7)
        for picture in pictures:
            self.assertTrue(picture._element.nvPicPr.cNvPr.get("descr"))

    def test_all_shapes_fit_inside_the_slide(self):
        for slide in self.deck.slides:
            for shape in slide.shapes:
                self.assertGreaterEqual(shape.left, 0, shape.name)
                self.assertGreaterEqual(shape.top, 0, shape.name)
                self.assertLessEqual(shape.left + shape.width, self.deck.slide_width + 2)
                self.assertLessEqual(shape.top + shape.height, self.deck.slide_height + 2)

    def test_zip_xml_and_relationships_are_valid(self):
        path = self.output / "Parking_Space_Reservation_Agent.pptx"
        self.assertEqual(validate_package(path)["slides"], 14)
        with ZipFile(path) as archive:
            self.assertIsNone(archive.testzip())
            for name in archive.namelist():
                if name.endswith((".xml", ".rels")):
                    ET.fromstring(archive.read(name))

    def test_native_text_boxes_do_not_overlap(self):
        for slide in self.deck.slides:
            boxes = [shape for shape in slide.shapes if shape.has_text_frame and shape.text]
            for index, first in enumerate(boxes):
                for second in boxes[index + 1 :]:
                    width = min(first.left + first.width, second.left + second.width) - max(first.left, second.left)
                    height = min(first.top + first.height, second.top + second.height) - max(first.top, second.top)
                    self.assertFalse(
                        width > 0.005 * 914400 and height > 0.005 * 914400,
                        f"Overlapping text on {slide.shapes.title.text}: {first.name} / {second.name}",
                    )

    def test_evaluation_covers_latency_as_well_as_accuracy(self):
        text = "\n".join(shape.text for shape in self.deck.slides[12].shapes if shape.has_text_frame)
        self.assertIn("5.574 s", text)
        self.assertIn("2026-10-31", text)

    def test_content_preserves_scope_and_reports_honest_metrics(self):
        text = "\n".join(shape.text for slide in self.deck.slides for shape in slide.shapes if shape.has_text_frame)
        for phrase in [
            "LangChain",
            "LangGraph",
            "Chroma",
            "SQLite",
            "stdio MCP",
            "First name",
            "Surname",
            "Vehicle identifier",
            "Duration",
            "Name",
            "Car Number",
            "Reservation Period",
            "Approval Time",
            "0.547",
            "0.500",
            "12/13",
            "90/90",
            "12/12",
            "391",
            "9 skipped",
            "not production capacity",
            "submission",
            "refusal",
        ]:
            self.assertIn(phrase, text)
        self.assertNotIn("TODO", text)
        self.assertNotIn("awaits replacement", text)

    def test_previews_exist_and_are_explicitly_not_office_renders(self):
        for number in range(1, 15):
            with Image.open(self.output / "previews" / f"slide-{number:02}.png") as image:
                self.assertEqual(image.size, (1920, 1080))
        self.assertTrue((self.output / "previews" / "contact-sheet.png").is_file())
        self.assertIn("not a PowerPoint/LibreOffice render", self.report["preview_method"])

    def test_bad_crop_and_overflow_fail_instead_of_silently_clipping(self):
        canvas = Canvas(Presentation(), "Bounds test", "TEST", 1, "README.md")
        with self.assertRaises(ValueError):
            canvas.picture("agent-diagram.png", 1, 2, 3, 3, (-1, 0, 10, 10))
        with self.assertRaises(ValueError):
            canvas.text("Text too tall", 1, 2, 1, 0.05, size=36)
        with self.assertRaises(ValueError):
            canvas.text("Outside slide", 14, 2, 1, 1)


if __name__ == "__main__":
    unittest.main()
