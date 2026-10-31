# Parking space reservation agent

A Python, LangChain, and LangGraph assignment project: a RAG chatbot collects
reservation details, a second agent coordinates human administrator approval,
and an MCP service records approved reservations in a text file.

**Scope:** implement the original four stages, not a production-grade parking
platform. Do not add scaling infrastructure, future-proof abstractions, or extra
features unless the user requests them. Required privacy, recording-service security,
reliability, evaluation, and testing remain part of the assignment.

- [PROJECT_REQUIREMENTS.md](PROJECT_REQUIREMENTS.md): fixed original assignment baseline,
  transcribed from the user's screenshots.
- [AGENTS.md](AGENTS.md): user intent, accepted choices, current status, and open decisions.
- [PARKING_DATA_MODEL.md](PARKING_DATA_MODEL.md): implemented fields and data boundaries.
- [RAG_EVALUATION.md](RAG_EVALUATION.md): retrieval/answer/latency evaluation report.
- [GUARDRAIL_EVALUATION.md](GUARDRAIL_EVALUATION.md): synthetic privacy-detector evaluation.

## Current status

| Stage | Implemented | Still needed |
| --- | --- | --- |
| 1: RAG/chatbot | Startup-ready Chroma/SQLite stores, 20-record fixture, graph-wired retrieval, three semantic eligibility-filtered options with location-preference terms and "other options" follow-ups, draft collection/validation/user confirmation, layered local privacy guards, synthetic guardrail evaluation, synthetic RAG retrieval/answer/latency evaluation and report | Optional retrieval tuning (see RAG_EVALUATION.md limits) |
| 2: Human approval | Second LangChain communication agent, explicit submission, SQLite period holds, typed Studio review, approval/refusal and user outcome replies; offline integration/race/replay/streaming tests; live extraction and sync/async administrator smoke checks; user-reported Studio demo | No remaining implementation for the selected Stage 2 scope; verification limits below |
| 3: Approved recording | File format and Python-owned recording responsibility selected | MCP service (or permitted fallback), access protection, approved-only file writes |
| 4: Orchestration | Conversation graph and offline integration tests | Complete approval/recording pipeline, end-to-end/load tests, completed usage/deployment documentation |

**Stages 1–2 are implemented; recording is not.** Confirmed details can be submitted
for a single offered space, held in SQLite, and approved or refused by a human in
Studio. User confirmation is not administrator approval; approval is not a successful
file write. MCP calls and approved-reservation file recording remain Stage 3 work.

## Setup and run

Use Python 3.12 or newer. Install the locked dependencies:

```bash
uv sync --dev
```

This also installs the pinned local `en_core_web_sm` English NLP model and
`presidio-analyzer`. Imports do not load NLP. Missing/broken privacy setup stops
startup; detection never downloads models or domain lists at runtime. The small CPU
model is supplemented by deterministic checks, not a comprehensive multilingual
detector. See [GUARDRAIL_EVALUATION.md](GUARDRAIL_EVALUATION.md).

Create a project-root `.env` from `.env.example`. Set:

- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `OPENAI_API_VERSION` for the
  current Azure-compatible/DIAL endpoint.
- `LANGSMITH_API_KEY` for Studio, and sign in to LangSmith in your browser.
- Optionally `AZURE_OPENAI_DEPLOYMENT` to override the existing DIAL model.

Start the development server from the project root:

```bash
.venv/bin/langgraph dev
```

Alternatively use `uv run langgraph dev`. In LangSmith Studio select
**parking_space_reservation**, switch to **Chat** mode, and send a message.
Continue in the same thread to retain conversation/draft state; create a new thread
for a separate conversation. **Graph** mode displays execution and internal state.

If Studio does not open automatically, use the URL printed by the CLI, normally
`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.
Local API docs: `http://127.0.0.1:2024/docs`. Stop the server with Ctrl+C.
`python app.py` is not a terminal chat interface. The selected interface is Studio;
an additional public UI is not part of the current scope.

The CLI loads `.env` through `langgraph.json`. Application imports do not load it,
initialize clients/databases, or make API calls. The configured Starlette lifespan
prepares databases and bounded baseline evidence **before the server accepts requests**.
The graph and HTTP app use module exports (`app:agent`, `app:webapp`) to share one agent
instance. Chat/extraction models remain lazy. First-time static seeding calls the
configured embedding provider; populated static collections are reused without seeding.
Startup failures stop the application with a stage-specific exception.
Agent Server manages thread checkpoints under `.langgraph_api/` (git-ignored).

### Studio connection troubleshooting

The printed API/Studio URLs do not mean application startup has completed. Starlette
supplies the custom startup/shutdown lifespan supported by Agent Server; it does not
add a separate public API or UI. Startup waits for both databases and baseline evidence.

- **Connect the VPN/network required by the configured Azure/DIAL endpoint before
  starting the server.** First-time Chroma seeding calls that endpoint for embeddings;
  an unreachable endpoint can leave startup waiting on request timeouts/retries and
  Studio showing “Failed to initialize Studio / Failed to fetch”. This failure was
  reproduced and resolved by connecting the VPN, without changing code. Keep the
  required connection active for model/embedding calls during chat, too.
- Check `http://127.0.0.1:2024/ok` and `/docs` in the same browser used for Studio. If
  they are unreachable, check server startup output and endpoint connectivity first.
  A populated Chroma collection needs no startup embedding call, but chat still may.
- If local API pages work but Studio cannot connect, allow **Local network access**
  for `https://smith.langchain.com` in Chrome's site settings, then reload Studio.
  This is a browser permission, not a reason to disable startup readiness or expose
  the server on all network interfaces. See
  [LangChain's Studio troubleshooting guide](https://docs.langchain.com/langsmith/troubleshooting-studio).

## Project structure and current workflow

| File/directory | Responsibility |
| --- | --- |
| `app.py`, `langgraph.json` | Shared CLI graph export and server-startup lifespan/configuration |
| `parking_agent.py` | Agent-owned configuration/resources and deterministic graph construction |
| `commons/constants.py` | Current model constants and user-facing system prompt |
| `context_augmentation.py` | Public-information retrieval and model-only evidence augmentation |
| `reservation_collection.py` | Draft extraction, validation, user confirmation, and cancellation |
| `administrator_agent.py` | Second LangChain agent: bounded review wording and strict human-decision schema |
| `storage/reservations.py` | Private SQLite hold ledger, atomic overlap checks, approval/refusal and timestamps |
| `guardrails.py`, `guardrail_evaluation.py` | Local privacy boundary checks and reproducible synthetic evaluation |
| `storage/` | Chroma static inventory, SQLite dynamic facts, configuration, and explicit seeding |
| `tests/` | Offline storage, graph, collection, and opt-in live-model checks |

Current graph:

```text
START → reservation_collection
  information turn → context_augmentation → parking_assitant → END
  incomplete request mentioning availability/reservation → context_augmentation → parking_assitant → END
  collection reply → parking_assitant → END
  successful /submit → parking_assitant (pending reply) → administrator_agent
    → administrator_review (Studio interrupt) → parking_assitant (outcome) → END
  invalid administrator response → administrator_review (fresh interrupt)
```

`parking_assitant` retains the user-selected spelling. The accepted complete
architecture uses two agents coordinated by Python/LangGraph, not a third orchestrator
LLM. The human administrator alone approves; Python owns allocation, reservation
state, approval enforcement, and file formatting/writing.

The agent owns shared resources, while conversation/request state belongs to graph
threads. Tests can inject models, extractors, clocks, embeddings, stores, and a
checkpointer. Direct callers outside Agent Server need a checkpointer for multi-turn
draft state; the public messages alone do not reconstruct it.
For standalone startup-ready use, construct with `require_startup=True`, call
`prepare_startup()` before graph invocation, and `finish_shutdown()` at shutdown.
The exported server graph always requires preparation; low-level standalone/offline
tests may leave `require_startup=False` to exercise individual components without
server-startup side effects.

## Persistent parking information

The selected static/dynamic split is an optional assignment improvement we are
keeping: local persistent Chroma for static information, built-in SQLite for hours,
prices, and availability. Neither requires a separate database server.

At each application-server startup:

1. Open the configured Chroma collection and **immediately seed** the bundled fixture
   if the collection is missing or empty. Reuse populated collections unchanged;
   do not compare timestamps, fixture contents, or freshness.
2. Recreate the configured SQLite file and schema empty, removing its old SQLite
   sidecars as well. This intentionally discards all dynamic facts **and private
   pending/approved/refused reservation records** from previous runs.
3. Read up to three scoped static records and initial dynamic facts, validate them,
   and prepare bounded baseline JSON evidence before accepting interaction. A missing
   configured location or unusable baseline is a startup error.

Preparation is idempotent within one server lifespan. New threads/graph invocations
do not reset SQLite or seed Chroma. The next server lifespan resets SQLite again.
**Do not share this disposable SQLite file between concurrently running servers.**
SQLite must be outside the persistent Chroma directory to prevent destructive path
overlap. The reset deletes only the configured database/sidecars, never their parent.
Personal reservation details and chats are not ingested into either public-information
store. Submitted details live in a separate private SQLite table, never public fact
CRUD or RAG evidence. Recording-directory cleanup is agreed for Stage 3, not implemented here.

### Static inventory and seeding

`StaticParkingStore` exposes startup `initialize`, scoped `list_space_ids`/`get_spaces`,
and the existing `upsert_documents`, `search`, and `delete_documents` APIs.
`ParkingSpace` validates individual inventory records and generates LangChain
documents. `SpaceFilters` supports exact space-ID filtering within mandatory
location/type/schema scope before semantic ranking.
The internal `search(..., eligible_ids=...)` constraint also limits ranking to the
document IDs Python has deemed eligible; it never changes that mandatory scope.

`storage/parking_spaces.json` contains 20 mock spaces for `demo-central`, differing
by identity and public directions. Every space fits any vehicle by project assumption;
no dimensions, vehicle categories, or facility-feature matching are modeled.
The fixture is not complete data for all required general parking/location questions.

Validate the fixture offline:

```bash
.venv/bin/python -m storage.seed --dry-run
```

Startup seeds automatically when needed. For an explicit operator-controlled upsert,
load `.env` and embed/upsert the spaces:

```bash
uv run --env-file .env python -m storage.seed
```

This calls the configured embedding provider. To use another reviewed inventory:

```bash
.venv/bin/python -m storage.seed --file path/to/reviewed-spaces.json --dry-run
uv run --env-file .env python -m storage.seed --file path/to/reviewed-spaces.json
```

Repeated upserts update the same IDs; unrelated IDs are preserved. Omitted records
are not deleted automatically. SQLite is untouched. The agreed demo uses the bundled
mock fixture without implementing static-data-change or approval logic. The explicit
seed CLI remains an operator API; startup never updates a populated collection.

**Expanding an existing eight-record demo:** stop the server, connect the VPN needed
by the embedding provider, run `uv run --env-file .env python -m storage.seed` once,
then restart. This preserves the original eight IDs and adds twelve records. Do not
delete Chroma or change collection/schema for this expansion. Editing the fixture
alone will not expand a populated collection. Unrelated IDs are preserved, so an
operator-maintained collection need not contain exactly 20.

**Existing database compatibility:** the current schema is version 2 and the default
collection is `parking_spaces_v2`. If `.env` selects an older collection, select a
fresh collection and re-seed. Do not upsert new-schema records over old-schema IDs:
Chroma merges metadata and can retain removed fields. Old-schema records are excluded
by search filters. Do not erase existing databases to migrate.
Changing embedding models also requires a fresh/reindexed collection. The current
short fixture needs no chunking; automatic chunking is not implemented.

### Dynamic facts

`DynamicParkingStore` exposes `upsert_fact`, `get_fact`, `list_facts`, and `delete_fact`.
Allowed kinds are `working_hours`, `prices`, and `availability`, keyed by location
and kind with JSON values and UTC write timestamps. Writes are transactional and
parameter-bound; connections are closed per operation.

Dynamic data starts empty at each server startup. Missing facts return `None` or an
empty list; hours and prices remain unknown until supplied through Python CRUD.
For availability suggestions, the static inventory is available by default unless a
successfully read dynamic snapshot blocks it. The current lightweight availability
snapshot supports a location-wide boolean or a map of space IDs to booleans:
`False` blocks a space; omitted IDs use the static default. Null, aggregate counts,
strings, and unsupported shapes cannot be used to select options and raise an error.
Hours/prices retain their existing JSON-compatible payloads.
These snapshots are public observations, not the private reservation ledger. A write
timestamp does not prove freshness. The Stage 2 ledger separately blocks overlapping
pending/approved periods and is never included in public evidence.

### Information retrieval

For information turns, code builds queries from fixed English public-topic terms,
never raw user text or draft values. Terms cover entrance, gate, lift, stairs, lobby,
pedestrian walkway, outdoor area, ground/first/second level, compass directions, wall,
fence, corner, location/directions, hours, prices, and availability. Location terms are
taken from the public inventory descriptions, so stated preferences affect ranking.
Request wording includes spaces/spots/places/slots, park (verb), free (except "free of
charge"), vacant, reserve/book, and options; "Tell me about parking" is information only. Unknown topics use `parking information` and reuse the bounded
startup static evidence. Location/directions topics still use semantic retrieval.
Hours/prices-only turns include only the requested operational facts, not space
inventory or unrelated availability snapshots. Dynamic facts
are reread on information turns, not taken from a permanently cached startup snapshot.

For availability/request topics, Python reads scoped inventory IDs, applies current
dynamic overrides, and uses the message-derived public-topic query for **embedding-based
similarity search over eligible spaces**. Chroma returns **up to three** records using
its native `k` limit. There is no random sampling of the whole inventory and no need
to implement extra space features: existing public descriptions provide searchable text.
For example, “Any available spaces near the elevator?” embeds `lift availability`,
not the full message or personal reservation details. Eligibility is applied before
ranking so blocked spaces do not displace eligible results. A failed read is never
an empty database. Retrieved records appear as `availability_options`, distinct from
ordinary static directions evidence. Offered IDs are retained in thread state, not
public graph input/output, and do not write or hold any space. Initial incomplete
reservation requests can display options alongside deterministic collection replies;
confirmation summaries and control commands retain their existing behavior.

**Other options.** A follow-up such as "Show me other options", "Any different spots?"
or "More spaces" excludes every space already offered in the current offer round,
before ranking, and returns up to three new eligible spaces. Exclusions come only from
code-tracked thread state (`offer_round_ids`), never from model text. A fresh request
without "other/more" starts a new round; `/cancel` and `/new` clear it. When nothing
remains, the evidence has an empty `availability_options` list and
`previous_offers_excluded` so the reply can say no further alternatives exist.
Blocked spaces are filtered before ranking, so fewer than three (or zero) options are
offered only when fewer eligible spaces remain, never because a blocked best match
displaced an eligible one.

For a validated duration, Python previews **now → now + requested hours** and excludes
overlapping pending/approved holds before ranking; otherwise it excludes only holds
active now and labels the evidence `current_observations`. Evidence carries the preview
UTC period and `eligibility_scope`, not
private ledger fields. `/options` explicitly requests suggestions using a code-owned
availability query. Offers are bound to `offer_duration_hours`; changing duration
invalidates them. Passing time alone does not invalidate confirmation or offers:
submission computes and atomically rechecks its own actual period.

JSON evidence is appended only to a copy of the latest user message for the model.
Dynamic snapshots have evidence-budget priority and are the instructed source for
operational facts. Missing/null/omitted hours/prices mean unknown; availability uses
the explicit code-controlled default/override rule above. Top-K retrieval is not
exhaustive inventory or allocation. Async storage runs off the event loop.

Current defaults: location `demo-central`, K=3, inspected user text=2,000 characters,
evidence=12,000 characters. `ContextConfig` holds these settings. Fixed queries narrow
free-form retrieval; follow-up rewriting and user-text exact-ID extraction are not
implemented. Model instructions are not enforced output grounding. Layered privacy
checks are implemented below, with documented detection limits.

### Storage settings

| Environment setting | Default |
| --- | --- |
| `PARKING_CHROMA_DIRECTORY` | `.data/chroma` |
| `PARKING_CHROMA_COLLECTION` | `parking_spaces_v2` |
| `PARKING_SQLITE_PATH` | `.data/parking.sqlite3` |
| `PARKING_LOCATION_ID` | `demo-central` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Existing `DIAL_TEXT_EMBEDDING_3_SMALL` constant |
| `AZURE_OPENAI_EMBEDDING_MODEL` | Existing `TEXT_EMBEDDING_3_SMALL` constant |

Relative paths are anchored to the project root. `.data/` is git-ignored.
Embedding clients initialize only when embedding is needed; empty searches do not
require them. Chroma telemetry is disabled.

## Reservation draft collection

The chatbot collects first name, surname, `car_number` (a user-provided vehicle
identifier, not necessarily a plate), and **how many hours** the user wants to reserve.
It does **not** ask for dates, start/end times, or the user's timezone. Reservations
are immediate, not scheduled for a future date. The application server's UTC clock
(`datetime.now(UTC)`) supplies the start when submission succeeds; there is no
LangSmith time lookup, extra service, or new dependency.

Use a positive number of hours, such as `2` or `1.5`; `hour(s)`, `hr(s)`, and `h`
suffixes are accepted. Common English expressions such as `two hours` or `half an hour`
also work. Unsupported or ambiguous expressions prompt for a numeric hour value;
Python never guesses a duration from dates or another unit. Nonfinite, zero, negative,
unrepresentably small, or overflowing durations are rejected.

Example synthetic message:

“Reserve parking. First name Demo, surname Tester, car number demo-bicycle-1.
I want to reserve for 2 hours.”

Python validates proposed verbatim field values and renders a revisioned summary.
Substring checks do not prove correct semantic extraction; the user reviews the summary.
`ValidatedReservationRequest` contains names, vehicle identifier, and numeric duration,
not a provisional start/end. `SubmittedReservationRequest` adds the canonical UTC period
inside the hold transaction. Thus time spent collecting details or choosing options
does not consume the requested duration. **Time waiting for administrator review after
submission does count**, as selected by the user; approval does not restart the clock.

Submitting without required details now starts/continues collection and names only the
missing fields (first name, surname, vehicle identifier, and how many hours). It explains
that the assistant will show the details for checking before sending a request to the
administrator, rather than showing the previous abstract confirmation/approval warning.
If all fields are supplied but not confirmed, `/submit` displays the confirmation
summary without creating a hold or implicitly confirming/submitting anything.

| Reply/command | Effect |
| --- | --- |
| `/confirm N` | Confirm displayed revision N |
| `yes`, `yes.`, `confirm`, `I confirm` | Confirm only immediately after the matching summary |
| `no`, `no.` | Decline details confirmation |
| A natural-language correction | Propose changes and validate/show a new summary |
| `/clear FIELD` | Remove `first_name`, `surname`, `car_number`, or `duration_hours` |
| `/cancel` | Clear the draft/confirmation, not chat history |
| `/new` | Start a new empty draft |
| `/options` | Show up to three semantic suggestions for a now-based duration preview; overlapping holds are excluded |
| `/submit SPACE_ID` | Submit one currently offered space for the confirmed duration, starting its pending hold now before administrator review |

States: `idle`, `collecting`, `awaiting_confirmation`, `ready`, `cancelled`.
Changes and rejected/unreadable updates invalidate confirmation. `ready` is collected,
user-confirmed information only. General information turns still use RAG while collecting.
Current commands/revisions/duration rules describe implemented behavior, not additional
assignment requirements that cannot be reconsidered during refactoring.

**Older Studio threads:** legacy draft schedules and confirmations are discarded on
read, while names/vehicle identifiers are retained. Provide hours and confirm the new
summary, or use `/new`/a fresh thread. Restarting still resets the disposable SQLite
ledger; old submitted checkpoints cannot resurrect a missing request. Chroma is unchanged.

## Stage 2: submission and human administrator review

### Demo in Studio

1. Start the server as above and use **Chat** mode to collect the required fields
   and confirm the displayed revision with `/confirm N`.
2. Send `/options`. Select **one of the IDs actually offered** using `/submit SPACE_ID`
   (for example, `/submit G-A01` only if that ID appears). Python rechecks eligibility
   and creates the pending hold before escalation. A conflict refreshes the options;
   choose again. Suggestions and confirmation alone never create holds.
3. The chatbot reports **pending** and the graph pauses at `administrator_review`.
   The human administrator switches the **same thread to Graph mode**, opens the
   interrupt, and reviews the request fields, space, and UTC period. This is the
   chosen local administrator channel—not email, Teams, or an external notification.
4. In the typed interrupt response form, review the reservation, select exactly
   `approve` or `refuse` in **Decision**, then click the right-hand **Resume** button.
   Do not use the lower-left graph-input **Submit** button for a human decision.
   No request ID needs to be copied, constructed, or entered. If a raw JSON editor
   is shown instead, supply only:

   ```json
   {"decision": "approve"}
   ```

5. The first agent's deterministic reply reports the actual SQLite result. Approval
   retains the allocation and UTC approval time; refusal releases the hold. Neither
   path writes a file. Return to Chat in the same thread to see the outcome. Use
   `/new` **after the decision** for a separate sequential one-space request.

The form has **one required field and no default decision**. Studio may immediately
show a required-field warning while **Decision** is unselected; this is validation
of the empty human response, not missing reservation details at handoff. Internal
`request_id`/draft IDs may still appear in private review/debug data, but the administrator
does not supply them. Python resolves the checkpointed submission in the current thread
and checks its review-payload identity against SQLite before showing/applying a decision.

**After updating from the older two-field form:** restart the local server and test a
fresh reservation. An already-paused interrupt can retain its old form schema in the
checkpoint. Startup deliberately clears SQLite, so old submitted requests become
unavailable; they are not migrated or resurrected. Persistent Chroma is untouched.

There is no new dependency, environment variable, REST route, public UI, or mail setup.
The second agent lazily uses the existing Azure/DIAL model with its own LangChain
structured-output runnable and administrator prompt. It receives only canonical
submitted details, not chat history, and generates a bounded neutral review message.
Model wording omits interface names and request facts; Python supplies the exact review
fields and UI instructions separately. This also avoids a local PERSON-detector false
positive on “Studio” in generated wording without weakening privacy checks.
Unsafe/unparseable wording or model API
failures use a fixed review request; wording never supplies decision authority.
Configuration/unsupported-model failures remain safe errors; the already-created
hold is retained and the failed graph run can be retried.

### Hold and decision rules

- Periods are half-open UTC intervals **[start, end)**. Pending and approved requests
  block overlaps on the same space/location, including earlier requests from the
  same thread. Adjacent periods can share a space. No multi-space request is modeled.
- `/submit` requires confirmed current details and a currently offered ID bound to
  that duration. After acquiring SQLite's write lock, Python captures current UTC time,
  derives end from the confirmed hours, rechecks public overrides and interval conflicts,
  and inserts the pending hold in the same short transaction. Transactions never span
  embeddings, model calls, or human review. Pending/outcome replies show the actual UTC period.
- The stable request ID is `draft_id:revision`. Replaying submission cannot create a
  duplicate or silently change its space/details. Administrator-node replay does not
  regenerate the review message; decision replay preserves the original timestamp.
- Only the typed administrator resume applies a decision. Chat claims, user `yes`,
  model wording, and forged message-input fields cannot approve a reservation. Missing,
  invalid, or extra response fields produce a fresh review interrupt without changing
  SQLite. A mismatched internal review/request identity fails with a clear exception
  before interrupting or applying any decision; the human never fixes this by entering an ID.
- While waiting, the current request cannot be edited, cancelled, or replaced by
  `/new`. It waits for human review without an expiry timer. After a decision, replies
  retain that outcome until `/new`; `/cancel` does not reverse an approved allocation.
- The fixed period starts at submission, and waiting for review counts toward its hours.
  Approval is allowed before end, even though start has already passed. If approval
  arrives at/after end, the request becomes `elapsed`, not approved, and its hold is released.
  Refusal is still allowed. Ended periods stop blocking on-demand queries without
  background cleanup or deleting historical records. Approval reversal is out of scope.
- SQLite is authoritative. A startup reset removes the ledger even if Agent Server
  still has an old checkpoint. An old review cannot resurrect a missing request:
  it reports the reset and asks for `/new`.
- Standalone callers must inject a checkpointer and use the same `thread_id` for
  `Command(resume=...)`; Agent Server supplies persistence automatically. Submission
  without a checkpointer is rejected before creating a hold.

### Trusted-local boundary and verification limits

**Studio and Agent Server remain trusted local development surfaces.** There is no
separate administrator login/role or multi-user authorization boundary. Anyone with
local graph resume/state access can act as the reviewer; automatic thread/request checks
correlate the response, not authenticate a human. Keep the server on localhost and give Studio
access only to the trusted operator. User and administrator roles are distinct workflow
steps, not distinct secured applications.

Necessary submitted personal fields reach the administrator-composition model and
appear in private interrupt payloads/SQLite/checkpoints. They do not enter Chroma,
public fact CRUD, or RAG evidence. Raw administrator model tokens are suppressed with
`nostream`; code-generated user pending/outcome replies appear in the `messages`
stream. Debug/state/task APIs, traces, and event streams remain trusted private data.
The general privacy limitations below still apply; this does not claim confidentiality
from the local developer or comprehensive prompt-injection protection.

Offline tests exercise actual sync/async LangGraph interrupts/resumes with temporary
Chroma/SQLite, injected models/embeddings/clocks, strict decision validation, overlap
races, refusal release, restart reset, graph replay after committed writes, and guarded
model streaming. A network-blocked real Agent Server test also checks the one-field
interrupt schema served to Studio, empty-response retry and decision-only approval.
On **2026-10-31**, four opt-in live smoke checks passed against the configured Azure/DIAL
endpoint: structured extraction, duration-only multi-turn collection/correction/confirmation,
and sync/async administrator structured output. The administrator checks inspect parsed
model output before fallback handling and verify that its wording is accepted while canonical
review facts remain unchanged. These checks use synthetic data and mocked stores, with tracing
disabled; approval/refusal graph integration remains covered by offline tests, not these
live smoke checks. The user also reported successful VPN-connected manual Studio testing
that day; individual browser scenarios were not itemized or independently checked by the
coding agent. This is smoke/demo evidence, not comprehensive live-model accuracy or security
evaluation.
Load tests and MCP/file integration remain Stages 3–4 work.

### Model compatibility

`PARKING_EXTRACTION_METHOD` defaults to `function_calling`; `json_schema` is an explicit
alternative. Actual endpoint support is required; there is no prose-parsing fallback.
Offline tests do not prove live compatibility. On 2026-10-31, all four opt-in synthetic
smoke checks passed for the current duration-only extraction and administrator schemas
with `function_calling` and `gpt-6-luna-2026-09-22` (VPN connected). No provider change
or static-data reset/reseeding was needed. See the verification evidence and limits above.
Live answer accuracy is measured separately in [RAG_EVALUATION.md](RAG_EVALUATION.md).

Opt-in extraction and sync/async administrator smoke tests (synthetic data, mocked stores,
no tracing; no pending hold or actual human decision is created):

```bash
LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false PARKING_LIVE_TEST=1 \
  uv run --no-sync --env-file .env python -m unittest discover -s tests -p test_reservation_model_live.py -v
```

## Required data protection and recording safety

Stage 1 privacy filtering now uses local Presidio/spaCy detection plus deterministic
credential patterns, known draft values, and labeled-value checks:

- Public static batches and dynamic JSON are checked before embeddings/writes.
  Unsafe batches are rejected entirely. Schema validation and trusted source review
  remain necessary; detection cannot identify every private value.
- Selected startup/turn evidence is checked before entering graph context/model
  prompts. Contaminated existing records cause safe failure, not automatic rewriting
  or collection-wide scanning. The static-reuse rule is unchanged.
- Necessary names, vehicle identifier, and duration hours still reach reservation extraction.
  Other detected sensitive data is replaced by offset-preserving spaces. Extracted
  updates must be verbatim in sanitized text, so removed values cannot become draft
  fields. Conflicting required values need clarification.
  PERSON spans are permitted for name extraction; this local filter cannot reliably
  distinguish unrelated names in that same message. Do not include unrelated names.
- General-answer calls use sanitized, text-only history copies, excluding message
  names, arbitrary metadata and caller-supplied system/tool messages. Collected draft
  names/vehicle identifiers stay in internal thread `privacy_values` for filtering
  later turns, even after corrections, `/cancel`, or `/new`. Neither command deletes
  chat/checkpoint history. Code-generated own-draft summaries remain visible.
- Generated responses are inspected and stripped of arbitrary model metadata/tool
  payloads. Failed checks return a fixed safe response. Unsupported non-text payloads
  are rejected rather than forwarded without inspection.
- Model calls use `nostream`: raw extraction/answer tokens are not emitted in the
  LangGraph `messages` stream before filtering. The final guarded response is emitted.
  This does not filter trusted debug/state/task APIs, `astream_events`, or traces.

English PERSON/common-PII detection, explicit secret patterns, and known values are
not complete confidentiality or prompt-injection protection. Locations/dates and
public prices are not blanket-redacted. Narrow structural rules avoid false positives
on generated space labels, technical numeric IDs, and currency amounts. Arbitrary
unlabeled vehicle identifiers and unfamiliar/obfuscated PII can be missed. See
[GUARDRAIL_EVALUATION.md](GUARDRAIL_EVALUATION.md) for results and limitations.

Stage 3 recording-service security/reliability remains unimplemented and required,
not a reason to build a production security platform.

Do not put secrets in chat. Inputs go to the configured model provider. Studio and
checkpoints expose private draft state; public schemas containing only messages do
not remove that data. The `.env.example` template disables tracing; existing `.env`
values are not changed. Enabled tracing can also store inputs/responses in LangSmith.
Filtering model copies does not sanitize stored chats or hide checkpoint state.
Keep the local development server bound to localhost while testing.

The future recording service must receive real human approval, not model-generated
authority. Pending/refused requests must not be recorded as approved. Required entry:
`Name | Car Number | Reservation Period | Approval Time`. Python owns the format;
approval and successful writing are separate results.

## Verification

Run the offline suite (temporary databases and deterministic embeddings, no API calls;
the installed local NLP model is used, with injected detectors for boundary tests;
live tests are skipped unless explicitly enabled):

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests use the standard-library `unittest` framework (the assignment accepts pytest or
unittest), so pytest is not a project dependency. If you install pytest yourself, it
can also discover and run these `unittest` test cases.

Run the synthetic privacy evaluation with `.venv/bin/python -m guardrail_evaluation`.
It prints case IDs/counts, coverage metrics, and warm detector latency; no files are
written.

Run the RAG evaluation (see [RAG_EVALUATION.md](RAG_EVALUATION.md)); the live CLI needs
the VPN and uses temporary stores only:

```bash
uv run --no-sync --env-file .env python -m rag_evaluation --part retrieval  # or --part all
```

Neither evaluation is a load test; load testing remains Stage 4 work.

Coverage includes Chroma/SQLite CRUD and persistence, validation/isolation/concurrency,
startup seeding/static reuse, SQLite reset/failure propagation, real Starlette lifespan,
eligibility-constrained semantic ranking/query privacy/thread isolation, context refresh/budgets/message preservation,
sync/async execution, and draft
collection/confirmation/duration handling, submission-time UTC periods, delayed review,
expiry boundaries, replay stability, public PII rejection, extraction redaction,
old-draft filtering, local-model readiness, guarded sync/async streaming, request/
location-term detection, "other options" exclusion, and RAG evaluation metrics/rules.
It does not establish complete privacy enforcement or completion of load tests.

Source checks:

```bash
uv tool run ty check
uv tool run ruff check app.py parking_agent.py administrator_agent.py guardrails.py guardrail_evaluation.py rag_evaluation.py reservation_collection.py context_augmentation.py commons storage tests
uv tool run ruff format --check app.py parking_agent.py administrator_agent.py guardrails.py guardrail_evaluation.py rag_evaluation.py reservation_collection.py context_augmentation.py commons storage tests
```

## Next steps

**Completed user-requested simplification:** immediate reservations using the server
UTC clock, collecting only duration hours instead of dates/timezone, and actionable
missing-details prompts instead of the confusing confirmation/approval warning.
The draft/offer/hold architecture and administrator expiry rules now support this flow.

Finish Stage 3 using the retained choices in AGENTS.md: decide MCP implementation
(or documented permitted fallback), output-file layout, retries, and practical access
protection. Preserve the implemented Stage 2 holds and decisions. If recording fails,
keep human approval and its applicable allocation in SQLite and explicitly report
the recording failure. Then finish full-pipeline integration/load tests and assignment
documentation in Stage 4. Do not add expiry timers or production infrastructure.

## References

- [LangGraph local server](https://docs.langchain.com/langsmith/local-server)
- [Server startup lifespan](https://docs.langchain.com/langsmith/custom-lifespan)
- [Studio setup](https://docs.langchain.com/langsmith/quick-start-studio)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph interrupts and typed Studio responses](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Using Studio](https://docs.langchain.com/langsmith/use-studio)
- [LangChain Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- [Python SQLite](https://docs.python.org/3/library/sqlite3.html)
