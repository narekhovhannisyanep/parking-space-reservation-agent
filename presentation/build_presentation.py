# /// script
# requires-python = ">=3.12"
# dependencies = ["python-pptx==1.0.2", "pillow==13.0.0"]
# ///
"""Build the assignment presentation without importing or modifying the app.

Run from the repository root: uv run --no-project presentation/build_presentation.py
Pillow previews mirror the authored layout; they are not Office-engine renders.
"""

import hashlib
import json
import math
import posixpath
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import PIL
import pptx
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

ASSETS = Path(__file__).resolve().parent
DECK_NAME = "Parking_Space_Reservation_Agent.pptx"
REPOSITORY = "https://github.com/narekhovhannisyanep/parking-space-reservation-agent"
WIDTH, HEIGHT, DPI = 40 / 3, 7.5, 144
BG = "0C111D"
PANEL = "151E2E"
LINE = "29374B"
WHITE = "F2F6FC"
MUTED = "ADBDD1"
TEAL = "53E0C1"
CYAN = "6CCBFF"
PURPLE = "B49AFF"
AMBER = "F7C46B"
CORAL = "F28E97"


@dataclass(frozen=True)
class SlideInfo:
    title: str
    section: str
    seconds: int
    source: str
    notes: str


SLIDES = [
    SlideInfo(
        "Parking Space\nReservation Agent",
        "PROJECT OVERVIEW",
        25,
        "PROJECT_REQUIREMENTS.md; README.md",
        "This project connects a parking-information chatbot to a human approval workflow "
        "and approved-only file recording. It is a Python learning assignment built around "
        "LangChain, LangGraph, RAG and MCP, rather than a production parking business. "
        "I will explain the responsibilities of each component, show the Studio workflow, "
        "and finish with the measured results and their limits. The central distinction "
        "throughout is that an LLM can help with language, but it cannot approve or allocate "
        "a reservation. Those transitions are explicit and controlled.",
    ),
    SlideInfo(
        "Four stages, one complete workflow",
        "ASSIGNMENT COVERAGE",
        35,
        "PROJECT_REQUIREMENTS.md, stages 1–4; AGENTS.md, accepted choices",
        "The original assignment has four stages. First, build a RAG chatbot that answers "
        "parking questions, collects the required details, filters sensitive data and has "
        "an evaluation report. Second, create a separate LangChain agent that communicates "
        "with the administrator. Third, record only human-approved reservations using MCP "
        "and the specified four-field text format. Fourth, integrate and test the workflow "
        "through LangGraph, including dialogue, review and recording load. Chroma, Studio "
        "review and a private stdio MCP server are the accepted implementation choices. "
        "Examples such as email or FastAPI were not additional mandatory features.",
    ),
    SlideInfo(
        "Two agents. Deterministic coordination.",
        "ARCHITECTURE",
        50,
        "README.md, project structure/current workflow; AGENTS.md, agent ownership",
        "There are two language-model agents, not a third orchestrator model. The parking "
        "agent handles user interaction and clarification. The administrator agent prepares "
        "a bounded review message from the canonical submitted request. Deterministic "
        "Python and LangGraph route the conversation, retrieve evidence, validate fields, "
        "create holds, apply human decisions and invoke recording. Chroma supplies public "
        "descriptions and hourly-price metadata. SQLite holds public dynamic facts and a "
        "separate private operational ledger. The recorder independently reads that ledger "
        "in read-only mode. Resources belong to the agent instance; conversation and request "
        "state belong to individual graph threads.",
    ),
    SlideInfo(
        "RAG uses public facts—not reservation data",
        "STAGE 1 / RETRIEVAL",
        40,
        "README.md, persistent parking information/information retrieval; PARKING_DATA_MODEL.md",
        "The public corpus contains twenty fictional parking spaces with differentiated "
        "route descriptions, benefits and trade-offs. Their fixed USD hourly prices are "
        "numeric Chroma metadata, not embedded prose. SQLite provides hours and availability; "
        "missing hours remain unknown. Its private reservation table is never public RAG "
        "evidence. Python derives a bounded public-topic query rather than embedding names, "
        "vehicle identifiers or raw chat. It filters eligibility before retrieving up to "
        "three similarity-ranked options and guards the evidence. The existing extraction "
        "call distinguishes parking from unrelated conversation, so unrelated turns skip "
        "retrieval rather than receiving stale parking context.",
    ),
    SlideInfo(
        "Filter first. Rank eligible options second.",
        "STAGE 1 / OPTIONS & BUDGETS",
        40,
        "README.md, optional hourly price constraints; RAG_EVALUATION.md, budgets/benefits",
        "Availability and price compliance are code responsibilities. Python excludes "
        "publicly blocked spaces and conflicting holds, then combines those eligible IDs "
        "with an optional numeric hourly-price filter in Chroma before native top-three "
        "ranking. Under eight dollars is strict; up to eight dollars includes eight; "
        "supported ranges include both ends. Total-stay costs, ambiguous units and unsupported "
        "currency cues require clarification rather than unrestricted offers. The budget is "
        "optional search state, not a fifth required reservation field. Descriptive preferences "
        "such as carrying bags affect semantic ranking, not guaranteed accessibility or "
        "feature suitability. Python displays every selected ID, description and rate.",
    ),
    SlideInfo(
        "Collect → validate → ask the user to confirm",
        "STAGE 1 / RESERVATION DETAILS",
        45,
        "README.md, reservation draft collection; presentation/confirming-user-info-question.png",
        "The four required fields are first name, surname, a user-provided vehicle identifier "
        "and duration in hours. The vehicle does not need a conventional plate format. "
        "This implementation deliberately uses immediate duration-only requests: it never "
        "asks the user for a date, start time or timezone. Model extraction proposes grounded "
        "values; Python validates them and owns the revisioned confirmation summary shown "
        "here. Explicit corrections invalidate confirmation, and a vague disagreement blocks "
        "submission until clarified. Saying yes confirms only the current details—it is "
        "neither a submission nor human administrator approval. Names and vehicle details "
        "shown in these user-approved screenshots belong to the demonstration.",
    ),
    SlideInfo(
        "The human administrator owns the decision",
        "STAGE 2 / HUMAN IN THE LOOP",
        45,
        "README.md, Stage 2 submission/review; presentation/interrupting-for-admin-review.png",
        "After confirmation, the user selects a currently offered space with slash-submit "
        "and its ID. Python captures the server UTC time inside an atomic SQLite hold "
        "transaction and calculates the end from the confirmed duration. The administrator "
        "agent then prepares the review wording, while Python supplies the exact canonical "
        "facts through a typed LangGraph interrupt. In the same Studio thread, the operator "
        "selects approve or refuse and clicks Resume. No request ID needs to be copied and "
        "there is no default decision. Waiting for review consumes the requested hours; "
        "approval at or after the fixed end becomes elapsed, not approved.",
    ),
    SlideInfo(
        "Only approval opens the recording gate",
        "STAGE 3 / PRIVATE STDIO MCP",
        45,
        "README.md, Stage 3 approved recording; PROJECT_REQUIREMENTS.md, file-entry format",
        "Python invokes exactly one narrow MCP tool, record_approved_reservation, using "
        "internal request and thread identifiers. No language model chooses an arbitrary "
        "write tool, filename, body or path. The private stdio server independently reads "
        "SQLite in read-only mode and verifies a real human-approved canonical request. "
        "It owns the required four-field UTF-8 format: full name, vehicle identifier, "
        "original UTC reservation period and original UTC approval instant. The format "
        "shown here is a template, not a screenshot of an output file. Publication is "
        "atomic and does not overwrite existing files; matching-content replay provides "
        "idempotent file effects. Only a validated success receipt permits a recording-success reply.",
    ),
    SlideInfo(
        "LangGraph makes each transition explicit",
        "STAGE 4 / ORCHESTRATION",
        45,
        "README.md, current graph/hold and decision rules; presentation/agent-diagram.png",
        "This is the actual Studio graph, including the intentionally retained spelling "
        "parking_assitant in code. Information turns retrieve evidence before answering. "
        "Collection, confirmation and submitted outcomes have code-controlled branches. "
        "After a successful submission, the graph emits the pending response, prepares "
        "administrator communication and pauses for review. Approval routes to recording; "
        "refusal or an elapsed request bypasses it. Invalid review values create a fresh "
        "interrupt without changing the ledger. Review preparation is checkpointed separately, "
        "so resuming does not create another hold or regenerate communication. Slash-new "
        "starts a separate one-space request only after a decision, preserving sequential workflow semantics.",
    ),
    SlideInfo(
        "Demo: browse, confirm, keep useful choices",
        "STUDIO WALKTHROUGH / USER",
        55,
        "README.md, option continuity; presentation/inital-question.png; presentation/confirming-user-info-answer.png",
        "The user starts by asking for parking and mentioning heavy bags. Early browsing "
        "is allowed, and the chatbot displays options while collecting the missing details. "
        "The user then provides the names, vehicle identifier and forty-eight-hour duration, "
        "and confirms the summary. Python automatically rechecks the earlier choices and "
        "keeps eligible IDs in their original order; the user does not need to issue a "
        "mandatory second slash-options search. This screenshot preserves OUT-V01, G-A02 "
        "and OUT-P01. The user explicitly submits OUT-V01. The heavy-bags example also "
        "illustrates a ranking limit: OUT-V01's own description warns that it is less "
        "convenient for bags. Retrieval is not perfect feature matching.",
    ),
    SlideInfo(
        "Demo: approval and refusal stay distinct",
        "STUDIO WALKTHROUGH / OUTCOMES",
        45,
        "presentation/admin-approved.png; presentation/admin-refuse.png; README.md, failures/retries",
        "These screenshots show the two administrator outcomes. Approval retains the "
        "allocation and original UTC period, then the MCP step returns a validated "
        "recording receipt and the chatbot reports recorded successfully. The refusal "
        "branch releases its pending hold and explicitly says that nothing was recorded. "
        "Human approval and file-writing success remain separate results: if recording "
        "fails, the approved allocation is retained and the reply must report failure or "
        "uncertainty, never pretend success. The selected policy allows one initial attempt "
        "and at most one transient retry, then a manual slash-retry-recording for the "
        "current approved request. There are no ordinary-chat or background retries.",
    ),
    SlideInfo(
        "Practical safeguards, explicit trust limits",
        "DATA PROTECTION & RELIABILITY",
        55,
        "GUARDRAIL_EVALUATION.md; README.md, data protection/recording safety",
        "Local Presidio, a pinned spaCy English model and deterministic rules guard public "
        "ingestion, selected evidence, model-copy history and generated replies. Necessary "
        "reservation fields can still reach the configured provider. Raw model tokens are "
        "suppressed from the guarded messages stream, but Studio, checkpoints and debug APIs "
        "remain trusted development surfaces. The recording service uses owner-only local "
        "permissions, approval checks, atomic publication and bounded subprocess sessions. "
        "The small synthetic privacy evaluation fully protected twelve of thirteen positive "
        "cases and falsely flagged none of eight benign cases. An unknown unlabeled vehicle "
        "identifier was missed. These are practical boundaries—not complete confidentiality, "
        "a sandbox or a separate authenticated administrator application.",
    ),
    SlideInfo(
        "Measured evidence—not production claims",
        "EVALUATION & SYSTEM VALIDATION",
        45,
        "RAG_EVALUATION.md, 2026-11-02; STAGE4_VALIDATION.md, historical load/current regression",
        "The priced twenty-query retrieval evaluation measured Recall-at-three of 0.547 "
        "and Precision-at-three of 0.500. Two live-answer runs passed twelve of thirteen "
        "conversations, and all twelve deterministic budget cases passed. The latest live-answer "
        "run's whole-turn median was 5.574 seconds (14 turns), not a latency guarantee. Results are mixed: "
        "exact-ID lookup and two challenge paraphrases remain weak spots. Separately, the "
        "historical full-workflow local load completed ninety conversations across concurrency "
        "one, five and ten, producing forty-five approved files. Those load models and review "
        "responses were injected or scripted, not provider or human throughput. Current "
        "HTTP/stdio regression covers thirteen scenarios; the latest documented offline "
        "suite has 391 passes and nine opt-in skips. Real-provider approval/refusal smoke "
        "is separate, small-sample evidence.",
    ),
    SlideInfo(
        "Four stages. One explicit approval flow.",
        "SUMMARY & PROJECT HANDOFF",
        30,
        "README.md, setup/deployment/repository links; PROJECT_REQUIREMENTS.md",
        "The delivered assignment slice combines RAG parking assistance, a second LangChain "
        "communication agent, human review, approved-only MCP recording and LangGraph "
        "orchestration. The takeaway is separation of responsibilities: models assist "
        "with language, the human decides, and Python enforces state transitions and writes. "
        "The repository contains setup instructions, architecture and evaluation reports. "
        "Run the locked environment locally, configure credentials, connect the required "
        "VPN and launch LangGraph Studio. Chroma persists, but SQLite and generated "
        "recording output deliberately reset on application startup. The repository link "
        "comes from project documentation; this presentation does not verify publication "
        "of the current local changes. Thank you—questions are welcome.",
    ),
]


def color(value):
    return RGBColor.from_string(value)


@lru_cache(maxsize=64)
def font(size, bold=False, mono=False):
    family = "Courier New" if mono else "Arial"
    match = subprocess.run(
        ["fc-match", "-f", "%{file}", family + (":style=Bold" if bold else "")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return ImageFont.truetype(match, round(size * DPI / 72))


def wrap_lines(text, width, size, bold=False, mono=False):
    """Explicit line breaks avoid relying on a recipient's automatic word wrapping."""
    selected = font(size, bold, mono)
    limit = width * DPI - 10
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            proposed = (line + " " + word).strip()
            if selected.getlength(proposed) <= limit:
                line = proposed
            else:
                if not line or selected.getlength(word) > limit:
                    raise ValueError(f"Word cannot fit at {size} pt: {word}")
                lines.append(line)
                line = word
        lines.append(line)
    return lines


class Canvas:
    """Small authoring helpers shared by native PowerPoint shapes and layout previews."""

    def __init__(self, prs, title, section, number, source):
        prs.slide_width, prs.slide_height = Inches(WIDTH), Inches(HEIGHT)
        self.slide = prs.slides.add_slide(prs.slide_layouts[5])
        self.image = Image.new("RGB", (1920, 1080), "#" + BG)
        self.draw = ImageDraw.Draw(self.image)
        self.number = number
        self.assets = set()
        self.text_boxes = []
        self.slide.background.fill.solid()
        self.slide.background.fill.fore_color.rgb = color(BG)
        self.rect(0.6, 0.36, 0.08, 0.15, TEAL)
        self.text(section, 0.79, 0.33, 11.94, 0.25, size=10.5, color_value=TEAL, bold=True)
        if "\n" not in title:
            self.text(title, 0.6, 0.79, 12.1, 0.65, size=32, bold=True, existing=self.slide.shapes.title)
        self.rect(0.6, 7.02, 12.1, 0.012, LINE)
        self.text(
            "Narek Hovhannisyan  /  Parking Space Reservation Agent", 0.6, 7.16, 7.7, 0.18, size=9, color_value=MUTED
        )
        self.text(f"{number:02} / 14", 11.75, 7.13, 0.95, 0.22, size=10, color_value=MUTED, align="right")
        for index in range(14):
            self.rect(9.55 + index * 0.14, 7.21, 0.09, 0.035, TEAL if index == number - 1 else LINE)
        self.text("Source: " + source, 0.6, 6.8, 12.1, 0.17, size=8.5, color_value=MUTED)

    def bounds(self, x, y, w, h):
        if min(x, y, w, h) < 0 or x + w > WIDTH + 0.001 or y + h > HEIGHT + 0.001:
            raise ValueError(f"Shape outside slide: {(x, y, w, h)}")

    def rect(self, x, y, w, h, fill=PANEL, border=None, rounded=False):
        self.bounds(x, y, w, h)
        shape = self.slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
            Inches(x),
            Inches(y),
            Inches(w),
            Inches(h),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = color(fill)
        if border:
            shape.line.color.rgb = color(border)
            shape.line.width = Pt(0.8)
        else:
            shape.line.fill.background()
        if rounded:
            shape.adjustments[0] = 0.08
        coords = (round(x * DPI), round(y * DPI), round((x + w) * DPI), round((y + h) * DPI))
        method = self.draw.rounded_rectangle if rounded else self.draw.rectangle
        kwargs = {"radius": round(min(w, h) * DPI * 0.08)} if rounded else {}
        method(coords, fill="#" + fill, outline="#" + border if border else None, **kwargs)
        return shape

    def text(
        self,
        content,
        x,
        y,
        w,
        h,
        size: float = 18,
        color_value=WHITE,
        bold=False,
        mono=False,
        align="left",
        existing=None,
        link=None,
    ):
        self.bounds(x, y, w, h)
        lines = wrap_lines(content, w, size, bold, mono)
        leading = size * 1.2
        if len(lines) * leading / 72 + 0.025 > h:
            raise ValueError(f"Text too tall on slide {self.number}: {content!r} ({len(lines)} lines, {h} in)")
        shape = (
            existing
            if existing is not None
            else self.slide.shapes.add_textbox(
                Inches(x),
                Inches(y),
                Inches(w),
                Inches(h),
            )
        )
        shape.left, shape.top, shape.width, shape.height = map(Inches, (x, y, w, h))
        shape.name = "Text: " + content.replace("\n", " ")[:70]
        tf = shape.text_frame
        tf.clear()
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.word_wrap = False
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = MSO_ANCHOR.TOP
        for index, line in enumerate(lines):
            paragraph = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
            paragraph.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
            paragraph.space_before = paragraph.space_after = Pt(0)
            paragraph.line_spacing = Pt(leading)
            run = paragraph.add_run()
            run.text = line
            run.font.name = "Courier New" if mono else "Arial"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color(color_value)
            if link:
                run.hyperlink.address = link
                run.font.underline = True
            length = font(size, bold, mono).getlength(line)
            px = x * DPI + (w * DPI - length) * {"left": 0, "center": 0.5, "right": 1}[align]
            self.draw.text(
                (round(px), round(y * DPI + index * leading * DPI / 72)),
                line,
                fill="#" + color_value,
                font=font(size, bold, mono),
                anchor="lt",
            )
        self.text_boxes.append((x, y, w, h, shape.name))
        return shape

    def arrow(self, x1, y1, x2, y2, accent=TEAL):
        self.bounds(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        shape = self.slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT,
            Inches(x1),
            Inches(y1),
            Inches(x2),
            Inches(y2),
        )
        shape.line.color.rgb = color(accent)
        shape.line.width = Pt(1.7)
        tail = OxmlElement("a:tailEnd")
        tail.set("type", "triangle")
        shape.line._get_or_add_ln().append(tail)
        start, end = (x1 * DPI, y1 * DPI), (x2 * DPI, y2 * DPI)
        self.draw.line((start, end), fill="#" + accent, width=3)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        points = [end] + [
            (end[0] - 12 * math.cos(angle + offset), end[1] - 12 * math.sin(angle + offset)) for offset in (-0.4, 0.4)
        ]
        self.draw.polygon(points, fill="#" + accent)

    def card(self, x, y, w, h, title, body, accent=TEAL, body_size=18):
        self.rect(x, y, w, h, rounded=True)
        self.rect(x + 0.24, y + 0.22, 0.38, 0.035, accent)
        self.text(title, x + 0.24, y + 0.40, w - 0.48, 0.75, size=22, bold=True)
        self.text(body, x + 0.24, y + 1.22, w - 0.48, h - 1.43, size=body_size, color_value=MUTED)

    def band(self, content, y=6.33, accent=TEAL, size=18):
        self.rect(0.6, y - 0.08, 12.1, 0.44, PANEL, rounded=True)
        self.rect(0.6, y - 0.08, 0.045, 0.44, accent)
        self.text(content, 0.78, y, 11.73, 0.34, size=size, color_value=accent)

    def picture(self, name, x, y, w, h, crop=None, description=None):
        self.bounds(x, y, w, h)
        with Image.open(ASSETS / name) as original:
            image = original.convert("RGB")
        if crop is not None:
            left, top, right, bottom = crop
            if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                raise ValueError(f"Invalid crop for {name}: {crop}, image {image.size}")
            image = image.crop(crop)
        factor = min(w / image.width, h / image.height)
        pw, ph = image.width * factor, image.height * factor
        px, py = x + (w - pw) / 2, y + (h - ph) / 2
        self.rect(px - 0.025, py - 0.025, pw + 0.05, ph + 0.05, PANEL, border=LINE)
        buffer = BytesIO()
        image.save(buffer, "PNG")
        buffer.seek(0)
        shape = self.slide.shapes.add_picture(buffer, Inches(px), Inches(py), Inches(pw), Inches(ph))
        shape.name = "Screenshot: " + name
        shape._element.nvPicPr.cNvPr.set(
            "descr",
            (description or name) + (f"; cropped region {crop}" if crop else "; full screenshot"),
        )
        resized = image.resize((round(pw * DPI), round(ph * DPI)), Image.Resampling.LANCZOS)
        self.image.paste(resized, (round(px * DPI), round(py * DPI)))
        self.assets.add(name)

    def subtitle(self, content):
        self.text(content, 0.6, 1.5, 12.1, 0.35, size=18, color_value=MUTED)


def author_slides(prs):
    canvases = []
    for number, info in enumerate(SLIDES, 1):
        # A short source label is visible; full citations stay in presenter notes.
        short_source = info.source.split(";", 1)[0].split(",", 1)[0]
        c = Canvas(prs, info.title, info.section, number, short_source)
        c.slide.notes_slide.notes_text_frame.text = (
            f"Suggested time: {info.seconds} seconds\n\n{info.notes}\n\nSources: {info.source}"
        )
        canvases.append(c)

        if number == 1:
            # The cover uses the title placeholder with a larger two-line layout.
            c.text(info.title, 0.6, 1.65, 7.8, 1.82, size=48, bold=True, existing=c.slide.shapes.title)
            c.text(
                "RAG assistance. Human approval.\nApproved-only recording.",
                0.64,
                3.75,
                7.2,
                1.08,
                size=25,
                color_value=MUTED,
            )
            c.text("Narek Hovhannisyan", 0.65, 5.55, 7.2, 0.5, size=24, color_value=TEAL, bold=True)
            c.text("Python  /  LangChain  /  LangGraph  /  MCP", 0.65, 6.14, 7.3, 0.4, size=18)
            c.rect(9.0, 1.72, 3.25, 3.95, PANEL, border=LINE, rounded=True)
            c.text("P", 9.55, 2.08, 2.13, 1.43, size=76, color_value=TEAL, bold=True, align="center")
            c.rect(9.54, 3.70, 2.13, 0.024, LINE)
            c.text("ASK\nREVIEW\nRECORD", 9.46, 4.0, 2.3, 1.12, size=20, bold=True, align="center")

        elif number == 2:
            c.subtitle("An assignment-scale solution, with a clear responsibility for every stage.")
            cards = [
                ("01", "RAG chatbot", "Parking answers\nDetail collection\nPrivacy + evaluation", TEAL),
                ("02", "Human review", "Second LangChain agent\nStudio interrupt\nApprove or refuse", PURPLE),
                (
                    "03",
                    "MCP recording",
                    "Approved-only writes\nFour-field text files\nPrivate + reliable",
                    AMBER,
                ),
                ("04", "LangGraph", "Integrated state flow\nEnd-to-end tests\nLocal load validation", CYAN),
            ]
            for index, (label, title, body, accent) in enumerate(cards):
                x = 0.6 + index * 3.09
                c.rect(x, 2.18, 2.83, 3.69, PANEL, rounded=True)
                c.text(label, x + 0.22, 2.43, 2.38, 0.6, size=34, color_value=accent, bold=True)
                c.text(title, x + 0.22, 3.18, 2.38, 0.85, size=22, bold=True)
                c.text(body, x + 0.22, 4.17, 2.38, 1.44, size=18, color_value=MUTED)
            c.band("Python + LangChain + LangGraph + RAG + a vector database.")

        elif number == 3:
            c.subtitle("Models assist with language; Python enforces the workflow and the human decides.")
            for x, title, body, accent in [
                (0.6, "Parking agent", "Answers parking questions\nCollects and clarifies user details", TEAL),
                (
                    7.0,
                    "Administrator agent",
                    "Prepares bounded review wording\nHuman chooses approve or refuse",
                    PURPLE,
                ),
            ]:
                c.rect(x, 2.1, 5.7, 1.72, PANEL, border=LINE, rounded=True)
                c.text(title, x + 0.23, 2.34, 5.22, 0.44, size=24, color_value=accent, bold=True)
                c.text(body, x + 0.23, 2.97, 5.22, 0.72, size=19, color_value=MUTED)
            c.arrow(6.38, 2.94, 6.87, 2.94, PURPLE)
            for index, (title, body, accent) in enumerate(
                [
                    ("Chroma", "Public descriptions\nFixed hourly-price metadata", TEAL),
                    ("SQLite", "Hours + availability\nPrivate hold ledger", CYAN),
                    ("stdio MCP", "Read-only approval checks\nApproved-only text files", AMBER),
                ]
            ):
                x = 0.6 + index * 4.13
                c.rect(x, 4.28, 3.84, 1.67, PANEL, rounded=True)
                c.text(title, x + 0.22, 4.52, 3.4, 0.44, size=24, color_value=accent, bold=True)
                c.text(body, x + 0.22, 5.15, 3.4, 0.65, size=18, color_value=MUTED)
            c.band("Azure/DIAL provides models + embeddings. Python + LangGraph coordinate; no third LLM.", size=17)

        elif number == 4:
            c.subtitle("A practical static/dynamic split keeps public evidence separate from private state.")
            c.card(
                0.6,
                2.08,
                5.89,
                3.21,
                "STATIC / CHROMA",
                "20 mock space descriptions\nRoutes, benefits and trade-offs\nUSD/hour prices as numeric metadata\nNo personal reservation details",
                TEAL,
            )
            c.card(
                6.81,
                2.08,
                5.89,
                3.21,
                "DYNAMIC / SQLITE",
                "Public hours and availability\nPrivate pending / approved holds\nPrivate ledger fields never enter RAG\nMissing hours remain unknown",
                CYAN,
            )
            for index, label in enumerate(
                ["Public-topic query", "Eligible IDs + filters", "Top-3 retrieval", "Guarded evidence"]
            ):
                x = 0.6 + index * 3.1
                c.rect(x, 5.64, 2.78, 0.48, PANEL, rounded=True)
                c.text(label, x + 0.07, 5.74, 2.64, 0.30, size=16, align="center")
                if index < 3:
                    c.arrow(x + 2.84, 5.88, x + 3.02, 5.88)
            c.band("Unrelated turns skip retrieval; only bounded public evidence reaches parking answers.", size=17)

        elif number == 5:
            c.subtitle("Hard availability and price rules are applied before semantic similarity ranking.")
            for index, (title, body, accent) in enumerate(
                [
                    (
                        "1 / Eligibility",
                        "Exclude public blocks\nExclude overlapping holds\nSuggestions do not hold spaces",
                        TEAL,
                    ),
                    (
                        "2 / Hourly budget",
                        "Validate explicit USD/hour cues\nFilter numeric Chroma prices\nNever pad over-budget results",
                        AMBER,
                    ),
                    (
                        "3 / Semantic ranking",
                        "Rank descriptions of eligible IDs\nReturn up to three options\nDisplay every ID and hourly rate",
                        PURPLE,
                    ),
                ]
            ):
                c.card(0.6 + index * 4.13, 2.12, 3.84, 2.98, title, body, accent, body_size=18)
            c.text("Optional budget—not a fifth reservation field", 0.65, 5.34, 12, 0.42, size=23, bold=True)
            c.text(
                "under $8/h: < $8     •     up to $8/h: ≤ $8     •     $5–$10/h: inclusive",
                0.65,
                5.86,
                12,
                0.35,
                size=19,
                color_value=AMBER,
            )
            c.band("Descriptive preferences rank options; they are not hard feature guarantees.", size=17)

        elif number == 6:
            c.subtitle("A checked draft is ready for selection—not submitted and not administrator-approved.")
            fields = ["First name", "Surname", "Vehicle identifier", "Duration in hours"]
            for index, label in enumerate(fields):
                y = 2.23 + index * 0.72
                c.rect(0.65, y, 4.64, 0.56, PANEL, rounded=True)
                c.text(f"0{index + 1}", 0.83, y + 0.1, 0.58, 0.33, size=18, color_value=TEAL, bold=True)
                c.text(label, 1.54, y + 0.09, 3.56, 0.36, size=20)
            c.text(
                "No dates or timezone questions.\nThe server clock starts the period\nwhen submission succeeds.",
                0.65,
                5.47,
                4.65,
                1.03,
                size=19,
                color_value=MUTED,
            )
            c.text("DEMO / USER DETAILS", 5.75, 2.13, 6.93, 0.28, size=12, color_value=TEAL, bold=True)
            c.picture(
                "confirming-user-info-question.png",
                5.76,
                2.55,
                6.9,
                0.72,
                (64, 75, 477, 99),
                "User supplies name, vehicle identifier and 48-hour duration",
            )
            c.picture(
                "confirming-user-info-question.png",
                5.76,
                3.56,
                6.9,
                2.06,
                (64, 317, 540, 445),
                "Code-generated revisioned confirmation summary",
            )
            c.text("User review, not human administrator approval", 5.75, 5.91, 6.93, 0.68, size=20, color_value=AMBER)

        elif number == 7:
            c.subtitle("The second LangChain agent prepares the request; a typed Studio interrupt pauses it.")
            c.text("01  /submit SPACE_ID", 0.65, 2.26, 5.1, 0.4, size=22, bold=True, color_value=TEAL)
            c.text(
                "Atomic eligibility recheck + pending hold\nServer UTC period starts at submission",
                0.65,
                2.88,
                5.1,
                0.72,
                size=19,
                color_value=MUTED,
            )
            c.text("02  Review canonical facts", 0.65, 3.98, 5.1, 0.4, size=22, bold=True, color_value=PURPLE)
            c.text(
                "Select approve or refuse → Resume\nNo default decision or request-ID copying",
                0.65,
                4.6,
                5.1,
                0.72,
                size=19,
                color_value=MUTED,
            )
            c.text("STUDIO / PRIVATE REVIEW PAYLOAD", 6.05, 2.13, 6.63, 0.28, size=12, color_value=PURPLE, bold=True)
            c.picture(
                "interrupting-for-admin-review.png",
                6.05,
                2.65,
                6.61,
                3.44,
                (26, 720, 480, 1054),
                "Pending request with canonical personal fields and UTC period",
            )
            c.band("Waiting counts toward the hours. Approval at/after the fixed end becomes elapsed.", size=17)

        elif number == 8:
            c.subtitle("One code-selected tool, no arbitrary path or body, and no network listener.")
            for index, (title, body, accent) in enumerate(
                [
                    ("Human-approved row", "SQLite is authoritative\nRecorder opens existing data read-only", PURPLE),
                    (
                        "Private stdio MCP",
                        "Verify status and canonical details\nTool: record_approved_reservation",
                        AMBER,
                    ),
                    ("Approved text file", "Atomic no-overwrite publication\nValidated success receipt", TEAL),
                ]
            ):
                x = 0.6 + index * 4.13
                c.rect(x, 2.2, 3.84, 2.2, PANEL, rounded=True)
                c.text(title, x + 0.21, 2.43, 3.42, 0.7, size=22, bold=True, color_value=accent)
                c.text(body, x + 0.21, 3.2, 3.42, 0.95, size=16, color_value=MUTED)
            c.text("REQUIRED FOUR-FIELD FORMAT / TEMPLATE", 0.65, 4.57, 12, 0.3, size=12, color_value=AMBER, bold=True)
            labels = ["Name", "Car Number", "Reservation Period", "Approval Time"]
            for index, label in enumerate(labels):
                x = 0.65 + index * 3.09
                c.rect(x, 5.06, 2.8, 0.8, PANEL, border=LINE, rounded=True)
                c.text(label, x + 0.08, 5.25, 2.64, 0.39, size=20, align="center")
                if index < 3:
                    c.text("|", x + 2.84, 5.27, 0.19, 0.4, size=20, color_value=AMBER, align="center")
            c.band("Full name • vehicle identifier • original UTC start/end • original UTC approval instant", size=17)

        elif number == 9:
            c.subtitle("Checkpointed review, explicit branches, and code-controlled outcome messages.")
            c.picture(
                "agent-diagram.png",
                0.65,
                1.99,
                5.25,
                4.63,
                description="Actual LangGraph diagram showing collection, retrieval, review and recording branches",
            )
            rows = [
                (
                    "COLLECT / ANSWER",
                    "Retrieve before information replies;\nconfirm and recheck current options.",
                    TEAL,
                ),
                ("SUBMIT / HOLD", "One currently offered ID;\natomic UTC overlap recheck.", CYAN),
                ("PAUSE / REVIEW", "Checkpoint the request;\nonly human resume applies a decision.", PURPLE),
                (
                    "RECORD / REPORT",
                    "Approve → MCP; refusal bypasses it.\nReport approval and recording separately.",
                    AMBER,
                ),
            ]
            for index, (label, body, accent) in enumerate(rows):
                y = 2.04 + index * 1.12
                c.rect(6.21, y + 0.03, 0.06, 0.87, accent)
                c.text(label, 6.5, y, 6.13, 0.31, size=14, color_value=accent, bold=True)
                c.text(body, 6.5, y + 0.4, 6.13, 0.68, size=19)

        elif number == 10:
            c.subtitle("Initial browse choices survive collection and confirmation when they remain eligible.")
            c.picture(
                "inital-question.png",
                0.68,
                2.02,
                11.97,
                0.65,
                (64, 72, 1067, 99),
                "Initial user request: parking with heavy bags",
            )
            c.text("BROWSE", 0.68, 2.95, 2.5, 0.4, size=21, color_value=TEAL, bold=True)
            c.arrow(2.6, 3.1, 3.43, 3.1)
            c.text("COLLECT", 3.78, 2.95, 2.5, 0.4, size=21, color_value=CYAN, bold=True)
            c.arrow(5.83, 3.1, 6.65, 3.1, CYAN)
            c.text("CONFIRM", 7.03, 2.95, 2.5, 0.4, size=21, color_value=PURPLE, bold=True)
            c.arrow(9.21, 3.1, 10.0, 3.1, PURPLE)
            c.text("SUBMIT", 10.4, 2.95, 2.25, 0.4, size=21, color_value=AMBER, bold=True)
            c.picture(
                "confirming-user-info-answer.png",
                0.68,
                3.68,
                11.97,
                2.21,
                (49, 395, 1071, 677),
                "Confirmed details with the same three offered space IDs and hourly prices",
            )
            c.band("No mandatory second /options search. The user explicitly selects /submit OUT-V01.", size=17)

        elif number == 11:
            c.subtitle("Only the approval branch invokes MCP; refusal releases the pending allocation.")
            c.text("APPROVED + RECORDED", 0.68, 2.12, 11.95, 0.42, size=23, color_value=TEAL, bold=True)
            c.picture(
                "admin-approved.png",
                0.68,
                2.75,
                11.97,
                1.25,
                (51, 609, 1090, 713),
                "Approval outcome retains the original period and reports a successful recording receipt",
            )
            c.text("REFUSED / NO RECORDING", 0.68, 4.37, 11.95, 0.42, size=23, color_value=CORAL, bold=True)
            c.picture(
                "admin-refuse.png",
                0.68,
                5.0,
                11.97,
                1.22,
                (43, 500, 1085, 613),
                "Refusal outcome explicitly releases the hold and says nothing was recorded",
            )
            c.band(
                "Approval ≠ successful file writing. On recording failure, retain approval and report the error.",
                size=17,
            )

        elif number == 12:
            c.subtitle("Assignment-appropriate protection for a trusted local Studio deployment.")
            c.card(
                0.6,
                2.12,
                3.84,
                3.99,
                "Guard boundaries",
                "Presidio + spaCy + rules\nFilter sensitive model-copy history\nGuard evidence and model output\nSuppress raw model tokens",
                TEAL,
            )
            c.card(
                4.73,
                2.12,
                3.84,
                3.99,
                "Protect recording",
                "Owner-only local files\nRead-only approval checks\nAtomic no-overwrite writes\nOne transient retry, then manual",
                AMBER,
            )
            c.card(
                8.86,
                2.12,
                3.84,
                3.99,
                "State the limits",
                "Necessary details reach provider\nStudio/checkpoints remain trusted\nDetection can miss unknown PII\nNo separate admin authentication",
                CORAL,
            )
            c.band("Synthetic privacy: 12/13 positive cases protected; 0/8 benign cases falsely flagged.", size=17)

        elif number == 13:
            c.subtitle("Dated, reproducible reports distinguish RAG quality, local load and small live smoke.")
            evidence = [
                (
                    0.6,
                    "RAG / 2026-11-02",
                    "0.547",
                    "Recall@3 / 20 queries",
                    "Precision@3: 0.500\nLive answer cases: 12/13\nBudget cases: 12/12\nLive turn median: 5.574 s",
                    TEAL,
                    "Small synthetic corpus; mixed results",
                ),
                (
                    4.73,
                    "LOCAL LOAD / 2026-10-31",
                    "90/90",
                    "Full-workflow conversations",
                    "Concurrency: 1 / 5 / 10\n45 approved output files\nInjected models; scripted review",
                    AMBER,
                    "not production capacity",
                ),
                (
                    8.86,
                    "REGRESSION / 2026-11-02",
                    "391",
                    "Offline tests passed",
                    "9 skipped (live opt-in)\n13 current HTTP/stdio cases\nReal-provider smoke: separate",
                    CYAN,
                    "Small live samples ≠ provider load",
                ),
            ]
            for x, label, value, caption, body, accent, limit in evidence:
                c.rect(x, 2.17, 3.84, 3.98, PANEL, rounded=True)
                c.text(label, x + 0.2, 2.4, 3.44, 0.31, size=11, color_value=accent, bold=True)
                c.text(value, x + 0.2, 2.97, 3.44, 0.8, size=46, color_value=accent, bold=True)
                c.text(caption, x + 0.2, 3.91, 3.44, 0.55, size=17, bold=True)
                c.text(body, x + 0.2, 4.55, 3.44, 1.17, size=17, color_value=MUTED)
                c.text(limit, x + 0.2, 5.83, 3.44, 0.24, size=10, color_value=accent)
            c.band("Known limits: exact-ID retrieval and two challenge paraphrases still miss.", size=17)

        elif number == 14:
            c.subtitle("RAG assistance → human review → approved-only MCP recording → explicit outcomes.")
            c.text(
                "Language belongs to models.\nDecisions belong to the human.\nEnforcement belongs to Python.",
                0.65,
                2.34,
                7.65,
                1.62,
                size=26,
                bold=True,
            )
            c.text("Project repository", 0.65, 4.57, 7.66, 0.56, size=24, color_value=TEAL, bold=True, link=REPOSITORY)
            c.text(
                "Setup and usage: README.md\nQuality and limits: RAG_EVALUATION.md\nIntegration/load: STAGE4_VALIDATION.md",
                0.65,
                5.37,
                7.67,
                1.1,
                size=19,
                color_value=MUTED,
            )
            c.rect(8.83, 2.23, 3.86, 3.98, PANEL, border=LINE, rounded=True)
            c.text("RUN LOCALLY", 9.05, 2.51, 3.42, 0.37, size=19, color_value=CYAN, bold=True)
            c.text(
                "uv sync --locked --dev\n\nConfigure .env\nConnect required VPN\n\n.venv/bin/langgraph dev",
                9.05,
                3.18,
                3.42,
                1.66,
                size=14,
                mono=True,
            )
            c.text(
                "Linux / localhost / Studio\nSQLite + output reset on startup",
                9.05,
                5.34,
                3.42,
                0.6,
                size=15,
                color_value=MUTED,
            )
    return canvases


def validate_package(path):
    """Check OPC relationships and XML independently of the authoring objects."""
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt ZIP member")
        names = set(archive.namelist())
        if len(names) != len(archive.namelist()):
            raise ValueError("Duplicate ZIP member")
        xml_count = 0
        rel_count = 0
        for name in names:
            if name.endswith((".xml", ".rels")):
                root = ET.fromstring(archive.read(name))
                xml_count += 1
                if name.endswith(".rels"):
                    base = "" if name == "_rels/.rels" else name.rsplit("/_rels/", 1)[0]
                    ids = set()
                    for relationship in root:
                        rid = relationship.attrib["Id"]
                        if rid in ids:
                            raise ValueError(f"Duplicate relationship ID in {name}")
                        ids.add(rid)
                        rel_count += 1
                        if relationship.attrib.get("TargetMode") == "External":
                            if relationship.attrib["Target"] != REPOSITORY:
                                raise ValueError("Unexpected external dependency")
                            continue
                        target = posixpath.normpath(posixpath.join(base, relationship.attrib["Target"]))
                        if target.lstrip("/") not in names:
                            raise ValueError(f"Broken relationship: {name} -> {target}")
        slides = [name for name in names if name.startswith("ppt/slides/slide") and name.endswith(".xml")]
        notes = [name for name in names if name.startswith("ppt/notesSlides/notesSlide") and name.endswith(".xml")]
        if len(slides) != 14 or len(notes) != 14:
            raise ValueError("Expected fourteen slides and fourteen notes parts")
        return {"slides": len(slides), "notes": len(notes), "xml_parts": xml_count, "relationships": rel_count}


def build_deck(output=ASSETS):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    preview_dir = output / "previews"
    preview_dir.mkdir(exist_ok=True)
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in ASSETS.glob("*.png")}
    prs = Presentation()
    props = prs.core_properties
    props.title = "Parking Space Reservation Agent"
    props.subject = "Four-stage LangChain/LangGraph RAG, human review and approved-only MCP workflow"
    props.author = "Narek Hovhannisyan"
    props.keywords = "parking, RAG, LangChain, LangGraph, human-in-the-loop, stdio MCP"
    props.comments = "14-slide English dark-theme assignment presentation with speaker notes."
    canvases = author_slides(prs)
    path = output / DECK_NAME
    prs.save(path)
    checks = validate_package(path)
    reopened = Presentation(path)
    for slide in reopened.slides:
        if not slide.shapes.title or not slide.has_notes_slide:
            raise ValueError("Missing accessible title or speaker notes")
        for shape in slide.shapes:
            if (
                shape.left < 0
                or shape.top < 0
                or shape.left + shape.width > reopened.slide_width + 2
                or shape.top + shape.height > reopened.slide_height + 2
            ):
                raise ValueError(f"Off-slide shape: {shape.name}")
    used = set().union(*(canvas.assets for canvas in canvases))
    if used != set(source_hashes):
        raise ValueError(f"Expected all seven approved screenshots, got {used}")
    for name, digest in source_hashes.items():
        if hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source image modified: {name}")
    notes = [
        "# Speaker notes — Parking Space Reservation Agent\n",
        "Author: Narek Hovhannisyan. English, 14 slides, suggested total: 10 minutes.\n",
        "Notes are also embedded in the PowerPoint. Timings are suggestions, not automatic slide advances.\n",
        "Screenshots are user-approved presentation assets. Crops preserve the originals; the format diagram is a template, not output-file evidence.\n",
    ]
    elapsed = 0
    sheet = Image.new("RGB", (1984, 1936), "#" + BG)
    sheet_draw = ImageDraw.Draw(sheet)
    for index, (canvas, info) in enumerate(zip(canvases, SLIDES, strict=True)):
        start, end = elapsed, elapsed + info.seconds
        notes.append(
            f"\n## {index + 1:02}. {info.title.replace(chr(10), ' ')}\n\n"
            f"**Suggested time:** {info.seconds}s ({start // 60}:{start % 60:02}–{end // 60}:{end % 60:02})\n\n"
            f"{info.notes}\n\n**Sources:** {info.source}\n"
        )
        elapsed = end
        canvas.image.save(preview_dir / f"slide-{index + 1:02}.png")
        x, y = 16 + (index % 3) * 656, 16 + (index // 3) * 384
        sheet.paste(canvas.image.resize((640, 360), Image.Resampling.LANCZOS), (x, y))
        sheet_draw.text((x, y + 361), f"{index + 1:02} / {info.section}", font=font(7), fill="#" + MUTED)
    sheet.save(preview_dir / "contact-sheet.png")
    (output / "SPEAKER_NOTES.md").write_text("\n".join(notes), encoding="utf-8")
    report = {
        "artifact": DECK_NAME,
        "author": props.author,
        "planned_seconds": elapsed,
        "slide_count": len(reopened.slides),
        "source_sha256": source_hashes,
        "package_checks": checks,
        "tools": {"python-pptx": pptx.__version__, "Pillow": PIL.__version__},
        "preview_method": "Pillow layout mirror; not a PowerPoint/LibreOffice render",
        "office_engine_render_verified": False,
        "application_dependencies_changed": False,
    }
    (output / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_deck(), indent=2))
