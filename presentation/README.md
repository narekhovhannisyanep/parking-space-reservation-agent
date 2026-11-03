# Project presentation

## Open the deck

**[Parking_Space_Reservation_Agent.pptx](Parking_Space_Reservation_Agent.pptx)**

- English, dark theme, 16:9 widescreen, 14 slides.
- Author: **Narek Hovhannisyan**.
- Suggested talk length: approximately **10 minutes**; timings total 600 seconds.
- Explanatory text and diagrams are editable PowerPoint shapes, not flattened slides.
- Presenter notes are embedded on every slide and exported to [SPEAKER_NOTES.md](SPEAKER_NOTES.md).
  Timings are suggestions; slides do not advance automatically. Use the notes as talking
  points rather than requiring a word-for-word reading.
- A clickable project-repository link is on the final slide. The link comes from the
  project README; publication/accessibility of the current local changes was not verified.

## Demo video

**[Watch or download demo.mp4](demo.mp4)**

The user-provided demo recording is available alongside the PowerPoint. Open the link
or download the MP4 and play it locally if your viewer does not support playback.
Viewing the recording does not require Python, provider credentials, or a running
application server. The video is a separate presentation asset, not embedded in the deck.
For reproducing the workflow, see the main README's [Studio demo instructions](../README.md#demo-in-studio).

Adding the recording does not change the separately documented evaluation results
or validation limits.

## Slide coverage

| Slides | Topic |
| --- | --- |
| 1–2 | Project goal and the assignment's four stages |
| 3 | Two-agent architecture, Chroma, SQLite, Azure/DIAL and private stdio MCP |
| 4–6 | RAG evidence, eligibility/optional budget filtering, collection and confirmation |
| 7 | Human administrator review and submission-time UTC holds |
| 8 | Approved-only MCP recording and the required four-field format |
| 9 | LangGraph orchestration and explicit decision branches |
| 10–11 | Studio demo: browse/confirm/submit, approval/recording and refusal |
| 12 | Sensitive-data protection, recording safeguards and trusted-local limits |
| 13 | Dated retrieval/latency results, historical local load and current regressions |
| 14 | Summary, local run instructions and project/documentation links |

The presentation distinguishes user confirmation, submission, human approval and successful
recording. It does not claim perfect preference matching, complete confidentiality,
administrator authentication or production/live-provider load capacity. Historical load
is not presented as a new benchmark. Evaluation figures come from the repository reports;
creating this deck did not rerun the application's model or load evaluations.

## Screenshot assets

The user removed the bug-related screenshots after those issues were solved and approved
**all seven current source PNG files in this directory for PowerPoint use**. They are
presentation assets, not diagnostic bug evidence. The builder uses all seven and verifies
their SHA-256 hashes before/after building. Originals remain unchanged; embedded crops
focus on relevant text. Screenshot pictures have descriptive alternative text.

The four-field recording diagram is explicitly a **format template**, not a screenshot
of an actual output file. The approval image shows the application's successful-recording
reply. Necessary demo details are visible in the user-approved screenshots.

## Previews and validation

[Contact sheet](previews/contact-sheet.png) and individual `previews/slide-01.png` through
`slide-14.png` are **Pillow layout mirrors**, not PowerPoint/LibreOffice renders.

Ten offline presentation-specific checks passed, covering:

- Fourteen slides in 16:9, correct author and unique title placeholders.
- Fourteen embedded notes parts and exported notes, with 600 seconds of suggested timing.
- All seven approved source images used without modification, with picture alt text.
- Slide bounds, nonoverlapping native text boxes, and fail-fast invalid crop/overflow handling.
- ZIP integrity, well-formed XML and resolved internal package relationships.
- Core workflow/scope statements, dated metrics and latency reporting.
- Preview dimensions and truthful distinction from Office-engine rendering.

The layout previews were visually reviewed, and the file was reopened with `python-pptx`.
No PowerPoint/LibreOffice renderer is installed in the build environment, so native
application rendering was **not independently verified**. Font substitution and rendering
can vary by machine; an operator review in PowerPoint is advisable before presenting.
This is not an uncompleted core assignment stage or a new system-validation claim.

[BUILD_REPORT.json](BUILD_REPORT.json) records package checks, tool versions, source
hashes and the rendering limitation.

## Rebuild or test

The user approved **isolated** `python-pptx` 1.0.2 and Pillow 13.0.0 tooling. The builder
declares these pins in PEP 723 script metadata. They are not application runtime/dev
dependencies and do not change `pyproject.toml`, `uv.lock`, the project `.venv`, `.env`,
or parking databases. Viewing the completed PowerPoint requires neither Python package.

From the repository root:

```bash
uv run --no-project presentation/build_presentation.py

uv run --no-project --with python-pptx==1.0.2 --with pillow==13.0.0 \
  python -m unittest discover -s presentation -p test_presentation.py -v
```

Rebuilding overwrites only the generated PowerPoint, notes, build report and named previews
in `presentation/`; it does not alter source screenshots. Tests generate into a temporary
directory. The Linux builder uses the existing `fc-match` command to locate Arial/Courier
compatible fonts (Liberation fonts in the validated environment). Initial isolated package
resolution needs internet access; cached tools can be reused afterwards.

For text/layout edits, modify `build_presentation.py` and rebuild. For ordinary edits,
open the `.pptx` in PowerPoint; a rebuild replaces those manual changes.
