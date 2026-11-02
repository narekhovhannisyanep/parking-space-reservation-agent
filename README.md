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
- [STAGE4_VALIDATION.md](STAGE4_VALIDATION.md): full-system integration, local load and live pipeline smoke report.

## Current status

| Stage | Implemented | Still needed |
| --- | --- | --- |
| 1: RAG/chatbot | Startup-ready Chroma/SQLite stores, 20 detailed mock descriptions with USD/hour metadata, optional validated persistent budgets, three availability/price-filtered semantic options with deterministic priced entries, draft confirmation, privacy guards, and measured retrieval/answer evaluation | Optional semantic retrieval tuning (see RAG_EVALUATION.md limits) |
| 2: Human approval | Second LangChain communication agent, explicit submission, SQLite period holds, typed Studio review, approval/refusal and user outcome replies; offline integration/race/replay/streaming tests; live extraction and sync/async administrator smoke checks; user-reported Studio demo | No remaining implementation for the selected Stage 2 scope; verification limits below |
| 3: Approved recording | Narrow Python stdio MCP service, read-only approval checks, atomic per-request files, bounded/manual retries, safe outcome replies, private permissions and startup cleanup; real MCP and Agent Server integration tests | No remaining implementation for the selected Stage 3 scope; trusted-local limits below |
| 4: Orchestration | Complete graph, 13 current HTTP/server/stdio scenarios, historical 90-conversation local load, refreshed small-load/contention checks, and real Azure/DIAL screenshot-flow approval/refusal smoke | No implementation remaining for the selected slice; repository publication/submission is a separate operator action; see report limits |

**All four stages are implemented and validated for the selected assignment slice.** Confirmed details can be submitted
for a single offered space, held in SQLite, and approved or refused by a human in
Studio. User confirmation is not administrator approval; approval is not a successful
file write. Only SQLite-confirmed human approvals proceed to the MCP recording step;
the final reply distinguishes approval from recording success/failure.

## Setup and run

Use Python 3.12 or newer. Install the locked dependencies:

```bash
uv sync --locked --dev
```

Stage 3 uses the official `mcp` Python SDK (locked to 2.3.0), without CLI extras,
LangChain MCP adapters, Node.js, FastAPI, an HTTP listener, or a new API key.
The tested recording deployment is **Linux with a local POSIX filesystem**: it
requires owner-only permissions, directory-descriptor/no-follow operations,
atomic hard links, and symlink-resistant cleanup. Unsupported primitives fail
closed rather than silently weakening file safety.

The locked project dependencies also include the pinned local `en_core_web_sm` English NLP model and
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
prepares databases, recording output/MCP readiness, and bounded baseline evidence
**before the server accepts requests**.
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
| `pricing.py` | Bounded verbatim hourly-budget validation, exact-cent comparisons and rate formatting |
| `administrator_agent.py` | Second LangChain agent: bounded review wording and strict human-decision schema |
| `storage/reservations.py` | Private SQLite hold ledger, atomic overlap checks, approval/refusal and timestamps |
| `recording.py` | Read-only canonical approval checks, four-field formatting, safe atomic file publication and dedicated cleanup |
| `recording_server.py` | Private stdio MCP server exposing only `record_approved_reservation` |
| `recording_client.py` | Code-selected sync/async MCP calls, bounded sessions/retries and safe receipt/error validation |
| `guardrails.py`, `guardrail_evaluation.py` | Local privacy boundary checks and reproducible synthetic evaluation |
| `system_validation.py`, `STAGE4_VALIDATION.md` | Isolated full-system HTTP/stdio scenarios, local load, opt-in live pipeline smoke and measured report |
| `storage/` | Chroma static inventory, SQLite dynamic facts, configuration, and explicit seeding |
| `tests/` | Offline storage, graph, collection, and opt-in live-model checks |

Current graph:

```text
START → reservation_collection
  information turn → context_augmentation → parking_assitant → END
  incomplete request mentioning availability/reservation → context_augmentation → parking_assitant → END
  collection reply → parking_assitant → END
  successful confirmation → context_augmentation (recheck existing options) → parking_assitant → END
  successful /submit → parking_assitant (pending reply) → administrator_agent
    → administrator_review (Studio interrupt)
      approved → record_approved_reservation → parking_assitant (approval + recording outcome) → END
      refused / elapsed / unavailable → parking_assitant (outcome) → END
  invalid administrator response → administrator_review (fresh interrupt)
  /retry-recording on current approved failed/unconfirmed request → record_approved_reservation → parking_assitant → END
```

`parking_assitant` retains the user-selected spelling. The accepted complete
architecture uses two agents coordinated by Python/LangGraph, not a third orchestrator
LLM. The human administrator alone approves; Python owns allocation, reservation
state, approval enforcement, and file formatting/writing.

The agent owns shared resources, while conversation/request state belongs to graph
threads. Tests can inject models, extractors, clocks, embeddings, stores,
checkpointers, and MCP client factories. Direct callers outside Agent Server need
a checkpointer for multi-turn
draft state; the public messages alone do not reconstruct it.
For standalone startup-ready use, construct with `require_startup=True`, call
`prepare_startup()` before graph invocation, and `finish_shutdown()` at shutdown.
The exported server graph always requires preparation; low-level standalone/offline
tests may leave `require_startup=False` to exercise individual components without
server-startup side effects.

## Persistent parking information

The selected static/dynamic split is an optional assignment improvement we are
keeping: local persistent Chroma for descriptions and fixed mock hourly prices,
built-in SQLite for hours, availability, and private reservation holds. Neither requires
a separate database server. The user-approved pricing change replaces SQLite prices
with Chroma numeric metadata, not a second price source.

At each application-server startup:

1. Open the configured Chroma collection and **immediately seed** the bundled fixture
   if the collection is missing or empty. Reuse populated collections unchanged;
   do not compare timestamps, fixture contents, or freshness.
2. Recreate the configured SQLite file and schema empty with owner-only permissions,
   removing its old SQLite sidecars as well. This intentionally discards all dynamic facts **and private
   pending/approved/refused reservation records** from previous runs.
3. Drop/recreate the dedicated `approved-reservations/` output directory alongside
   SQLite, with owner-only permissions. Verify the read-only ledger/output setup and
   the MCP tool schemas through a bounded real stdio session. Recording-path checks
   run before any destructive reset; cleanup never deletes SQLite's parent or Chroma.
4. Read up to three scoped static records and initial dynamic facts, validate them,
   and prepare bounded baseline JSON evidence before accepting interaction. A missing
   configured location or unusable baseline is a startup error.

Preparation is idempotent within one server lifespan. New threads/graph invocations
do not reset SQLite or seed Chroma. The next server lifespan resets SQLite again.
**Do not share this disposable SQLite file between concurrently running servers.**
SQLite must be outside the persistent Chroma directory to prevent destructive path
overlap. The reset deletes only the configured database/sidecars, never their parent.
Personal reservation details and chats are not ingested into either public-information
store. Submitted details live in a separate private SQLite table, never public fact
CRUD or RAG evidence. Recording files are private approved-only outputs, never public
facts or another allocation source.

### Static inventory and seeding

`StaticParkingStore` exposes startup `initialize`, scoped `list_space_ids`/`get_spaces`,
and the existing `upsert_documents`, `search`, and `delete_documents` APIs.
`ParkingSpace` validates individual inventory records and generates LangChain
documents. `SpaceFilters` supports exact space-ID filtering within mandatory
location/type/schema scope before semantic ranking.
The internal `search(..., eligible_ids=...)` constraint also limits ranking to the
document IDs Python has deemed eligible; it never changes that mandatory scope.

`storage/parking_spaces.json` contains 20 mock spaces for `demo-central`, differing
by location, access routes, practical benefits/trade-offs, and fixed hourly price.
Each has required `hourly_price_cents` from 100 to 2000 ($1–$20/hour, USD). Numeric
prices are Chroma metadata, not embedded prose. Descriptions mention useful routes,
shopping/unloading, pushchairs, weather exposure and passing traffic; these are fictional
evidence, not certified accessible bays or hard feature guarantees. Every space fits
any vehicle; no dimensions, vehicle categories, or feature flags are modeled.
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

**Migrating the earlier demo to priced inventory:** stop the server and change only
`PARKING_CHROMA_COLLECTION` in your existing `.env` to `parking_spaces_v3` (the new
default and `.env.example` value). Connect the required VPN, then run
`uv run --env-file .env python -m storage.seed --dry-run` and
`uv run --env-file .env python -m storage.seed`, then restart. Alternatively an empty
v3 collection will seed at startup. The old Chroma collection/data is retained, and
the original 20 space IDs are preserved in the new collection. Do not erase Chroma
or upsert schema-3 records into an old-schema collection. Editing the fixture alone
never changes a populated collection. Your real `.env` and `.data/` are not modified
by the coding/evaluation work.

**Existing database compatibility:** the current schema is version 3 and the default
collection is `parking_spaces_v3`. If `.env` selects an older collection, select a
fresh collection and re-seed. Do not upsert new-schema records over old-schema IDs:
Chroma merges metadata and can retain removed fields. Old-schema records are excluded
by search filters. Do not erase existing databases to migrate.
Changing embedding models also requires a fresh/reindexed collection. The current
short fixture needs no chunking; automatic chunking is not implemented.

### Dynamic facts

`DynamicParkingStore` exposes `upsert_fact`, `get_fact`, `list_facts`, and `delete_fact`.
Operational kinds are `working_hours` and `availability`, keyed by location
and kind with JSON values and UTC write timestamps. Writes are transactional and
parameter-bound; connections are closed per operation.

Dynamic data starts empty at each server startup. Missing facts return `None` or an
empty list; hours remain unknown until supplied through Python CRUD. Prices come
only from scoped Chroma inventory metadata; a missing inventory has unknown rates.
For availability suggestions, the static inventory is available by default unless a
successfully read dynamic snapshot blocks it. The current lightweight availability
snapshot supports a location-wide boolean or a map of space IDs to booleans:
`False` blocks a space; omitted IDs use the static default. Null, aggregate counts,
strings, and unsupported shapes cannot be used to select options and raise an error.
Hours retain their existing JSON-compatible payloads. Legacy `prices` CRUD is kept
for API compatibility but ignored by RAG/filtering/display; do not populate it.
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
Hours/prices-only turns include requested hours and/or an aggregate Chroma hourly
price range read from scoped metadata, not space
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
collection preserves those choices and their code-owned descriptive query. A selected
space ID (including an `L1-`/`L2-` prefix) is not a new descriptive preference or a vehicle identifier.
After confirmation, Python automatically reloads/rechecks the existing choices and
retains eligible IDs in their original order. Only removed choices are replaced using
the existing public preference query; the reply identifies options that are no longer
eligible. If no options were shown before collection, confirmation retrieves them automatically.
There is no mandatory second `/options` search.

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
availability query. `/options` is optional and rechecks the same current choices;
"other options" deliberately requests a different set. Offers are bound to
`offer_duration_hours`; changing duration invalidates that binding and personal
confirmation, but preserves browse candidates/preferences. Fresh confirmation
automatically rechecks them for the new duration. Passing time alone does not invalidate confirmation or offers:
submission computes and atomically rechecks its own actual period.

JSON evidence is appended only to a copy of the latest user message for the model.
Dynamic snapshots have evidence-budget priority and are the instructed source for
hours/availability. Missing/null/omitted hours mean unknown; availability uses
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
| `PARKING_CHROMA_COLLECTION` | `parking_spaces_v3` |
| `PARKING_SQLITE_PATH` | `.data/parking.sqlite3` |
| `PARKING_LOCATION_ID` | `demo-central` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Existing `DIAL_TEXT_EMBEDDING_3_SMALL` constant |
| `AZURE_OPENAI_EMBEDDING_MODEL` | Existing `TEXT_EMBEDDING_3_SMALL` constant |

Relative paths are anchored to the project root. `.data/` is git-ignored.
Recording output uses the fixed sibling name `approved-reservations/` beside the
configured SQLite file: `.data/approved-reservations/` by default. An explicit SQLite
path override relocates that disposable output too, keeping isolated test deployments
out of the real project data. There is no chat-selected output path or additional
recording-path environment variable. Do not put backups or other valuable files there.
Embedding clients initialize only when embedding is needed; empty searches do not
require them. Chroma telemetry is disabled.

### Optional hourly price constraints

Give an explicit USD/hour upper limit or range, for example:

- “Any available spaces under $8/hour?” — strictly below $8.
- “Up to $8 per hour.” — $8 is included.
- “Between $5 and $10/hour.” — both ends included.
- “Budget: eight dollars an hour.” — up to $8/hour.

The existing structured extraction call proposes a complete verbatim `budget_phrase`;
Python independently validates recognizable price cues, currency, hourly units,
comparison and numeric bounds. An omitted model phrase cannot ignore supported
explicit wording. Ambiguous/fuzzy/total-stay/unsupported/conflicting budgets prompt
for clarification rather than generating unrestricted options. This is a bounded
English grammar, not universal extraction accuracy. Decimal amounts must be whole
cents; number words zero through twenty are supported. User budgets are not clamped
to the $1–$20 fixture range; a $0.50/hour cap returns no matches. Very large amounts
above $10,000,000/hour are unsupported.

Budget is optional and thread-scoped, separate from personal details/confirmation.
It persists through `/options` and “other options”; changes invalidate previous offers
and start a new offer round without changing confirmed personal details. `/clear-budget`
(or “clear my budget”) removes it; `/new` and `/cancel` also reset it. A price preference
awaiting clarification blocks new offers and submission until resolved/cleared.

Python combines current availability/hold eligibility with numeric Chroma price
filters **before** semantic top-3 ranking and the result-count check. It never samples
or pads the list with over-budget spaces. Every code-selected option is displayed
with its ID, description and `$X.XX/hour (USD)`, including collection/control replies.
Application code appends these entries to guarded model explanations; general model
prose remains prompt-grounded, not a second authority for options or prices.
Submission rereads the selected rate under an active budget and retains the existing
atomic SQLite overlap/public-availability recheck. Static prices are fixed reviewed
mock data, not locked quotes or a concurrent billing system. No total estimate,
payment processing, rate generator, new mandatory reservation field or MCP column is added.

Queries also map mobility/step-free access, shopping bags/unloading, pushchairs,
covered routes/rain, passing traffic and quick departures to fixed public terms.
“Avoid stairs” maps to step-free access rather than a positive stairs query. Richer
descriptions do not guarantee semantic matches; measured limits are in RAG_EVALUATION.md.

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
Occurrence-aware checks reject negated values and code-tracked space references/fragments
as vehicle updates. An explicitly labeled vehicle identifier may legitimately match a
space ID; there is no plate-format restriction. Structured extraction also proposes
verbatim field-rejection evidence or a vague-disagreement phrase, independently checked
by Python. Bounded English safeguards catch common rejection wording even if the model
omits it. These checks are not universal semantic understanding; review the summary.
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
| `no`, `no.` or “Those details are wrong” | Invalidate confirmation and ask which detail needs correction; `yes` cannot bypass unresolved disagreement |
| A natural-language replacement | Validate the positive replacement and show a new summary requiring fresh confirmation |
| “My vehicle id isn't OUT-P01” | Clear the rejected vehicle field, keep other details/options, and ask for the correct identifier |
| `/clear FIELD` | Remove `first_name`, `surname`, `car_number`, or `duration_hours` |
| `/cancel` | Clear the draft/confirmation, not chat history |
| `/new` | Start a new empty draft |
| `/clear-budget` | Clear the optional price preference and invalidate old offers; not the personal draft |
| `/options` | Optional: recheck/show current choices for a now-based duration preview; eligible choices retain order |
| “Other options” | Deliberately retrieve different eligible choices, excluding all IDs already shown in the round |
| `/submit SPACE_ID` | Submit one currently offered space for the confirmed duration, starting its pending hold now before administrator review |
| `/retry-recording` | Retry file recording for the current human-approved failed/unconfirmed request, before `/new`; does not repeat approval or shift the period |

States: `idle`, `collecting`, `awaiting_confirmation`, `ready`, `cancelled`.
Changes and rejected/unreadable updates invalidate confirmation. `ready` is collected,
user-confirmed information only. General information turns still use RAG while collecting.
Vague disagreement uses internal `correction_pending`; it preserves the personal draft
but blocks confirmation/submission until a concrete correction or field clear. Named
rejections clear only that field, and replacement-in-the-same-message uses only the
positive occurrence. Required missing fields always block submission. Previously collected
values remain in private filtering history; correction is not chat/checkpoint deletion.
Automatic revalidation failure leaves confirmed personal details but no eligible offered
IDs/duration binding; it reports an error, never submits stale choices. Retry the failed
run or use optional `/options` after restoring storage/provider availability.
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
2. Use the automatically rechecked options. Select **one of the IDs actually offered** using `/submit SPACE_ID`
   (for example, `/submit G-A01` only if that ID appears). Python rechecks eligibility
   and creates the pending hold before escalation. A conflict refreshes the options;
   choose again. `/options` is optional; "other options" requests alternatives.
   Suggestions and confirmation alone never create holds.
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
   retains the allocation and UTC approval time, then runs MCP recording; refusal
   releases the hold without recording. Return to Chat in the same thread to see
   both approval and recording outcomes. Use
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
MCP/file integration is covered by the Stage 3 tests below. The completed Stage 4
local load and live pipeline smoke evidence is separate in [STAGE4_VALIDATION.md](STAGE4_VALIDATION.md).

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

## Stage 3: approved recording through MCP

### Normal use

Continue the Stage 2 Studio demo above. After selecting **approve → Resume**, the
graph runs `record_approved_reservation` automatically—no extra human approval,
request-ID copying, shell command, or third LLM is needed. A successful final reply
says **Recorded successfully** and gives the service-generated `.txt` basename.
Inspect that file locally in `.data/approved-reservations/` (or the sibling of an
explicitly relocated SQLite database).

Each file contains one UTF-8 line with exactly the required four fields:

`First name Surname | Car Number | UTC start to UTC end | UTC approval time`

The server uses the original submission-time period and human approval instant,
not its current clock. Recording after the period has ended is still allowed if
human approval was valid before end. No space ID, request ID, extra column, or model
wording is added to the entry. Existing single-line field validation rejects pipes
and control characters; Unicode and non-plate vehicle identifiers are preserved.

### Service and access boundary

- The official Python SDK implements real MCP over private subprocess stdin/stdout.
  The user selected this instead of an HTTP/FastAPI service; the permitted direct
  function-call fallback is **not** used. Python graph code selects the tool, not an LLM.
- The only tool accepts `request_id` and `thread_id`. It accepts no approval flag,
  reservation body, filename, directory, arbitrary text, or arbitrary-write command.
  Unknown tools/extra fields are rejected. The human still reviews only Decision in Studio.
- The server independently opens existing SQLite with `mode=ro`, loads the matching
  private row, checks actual `approved` status and canonical field/time consistency,
  and formats the record itself. Missing/corrupt/pending/refused/elapsed records cannot
  be written; a missing database is never initialized or reconstructed by the recorder.
- Output names are SHA-256 hashes of internal request identities. Data is flushed
  and synchronized in a private temporary file, then atomically published without
  overwriting any existing target. Matching replay succeeds; conflicting contents,
  unsafe links, and nonregular targets fail. Replay checks bounded output bytes only
  for deduplication, never for allocation or syncing the ledger from files.
- Output directories are `0700`; final/temporary files and SQLite are `0600`.
  Cleanup is directory-descriptor scoped and rejects unsafe paths/storage overlaps.
  A temporary-name collision never deletes another writer's temporary file.
- Each attempt launches the fixed Python script without a shell, opens a short-lived
  MCP session, and closes it. Provider keys are not passed to the child; it does not
  load `.env`. Stdout is reserved for MCP. Receipts contain bounded identity/status/
  filename/error codes, not personal fields or raw exception/protocol text.

This is practical **trusted-local** protection, not multi-user administrator
authentication or an OS sandbox. Thread matching correlates a request, not authenticates
a human. The trusted account/root can read or alter its private data and code; Studio,
checkpoints, development APIs, and optional parent tracing retain their existing private
boundaries. Do not expose the development server or recording data to untrusted users.

### Failures and retries

Approval is committed before recording, with no SQLite write transaction spanning
MCP/file I/O. The recorder never changes approval or allocation.

| Recording outcome | User-facing behavior |
| --- | --- |
| `recorded` | Validated MCP receipt confirms successful recording; report the basename |
| `failed` | Explicit failure; keep approval/allocation and show retry instructions |
| `unconfirmed` | Communication/acknowledgment or post-publication failure; a file may exist, so never claim success or claim nothing was written |

The client makes an initial attempt and **at most one retry** for recognized transient
transport/storage failures. Permanent failures and invalid receipts are not automatically
retried. Each attempt has a 15-second deadline plus bounded SDK shutdown/cleanup time;
processes and diagnostic handles are closed even on timeout/cancellation. Cancellation
propagates, and unexpected programming errors are not hidden as transient failures.

After failure/unconfirmed output, fix the underlying issue and send `/retry-recording`
in the **same thread before `/new`**. It reloads the current approved request and repeats
only recording. Already recorded requests return their existing outcome without another
write; pending/refused/elapsed/missing requests cannot use this command to gain approval.
Ordinary chat does not repeatedly retry. There is no background retry worker, approval
reversal, or change to submission/end/approval timestamps. A pre-recording ledger read
failure explicitly reports that file writing was not attempted in that step; it does not
incorrectly say that no administrator decision was applied.

Startup intentionally discards both the ledger and generated outputs. Old checkpoints
cannot resurrect a reset reservation. `/new` clears the current recording receipt and
starts a separate request; it does not reverse earlier allocations or repair old outputs.

### Stage 3 verification

The offline suite includes 16 writer tests, 11 MCP protocol/client tests and 9 recording
graph tests. It exercises actual stdio success/replay, stalled-process timeout and native
cancellation cleanup, strict schemas and unsafe responses (including SDK exception groups),
concurrent same/different request writes, no partial final entries, collision/link safety,
failure before/after publication, manual retry, unchanged approval/allocation, Unicode/UTC
rendering and startup reset. Most graph fixtures use real in-process MCP with injected
agent models to avoid spawning a subprocess for every fixture.

The network-blocked real Agent Server test now follows collection → hold → decision-only
Studio-compatible resume → actual stdio MCP → exact recorded file and final reply. This
uses synthetic data and injected models, not a browser/live-provider accuracy test.
The full run on 2026-10-31 completed **319 tests, with 4 opt-in live checks skipped**;
focused checks were also rerun during implementation. Concurrency tests are not the
assignment's full-system load test.

Run just the recording checks:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_recording*.py' -v
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

Stage 3 adds the narrowly scoped, owner-protected approved-recording service above;
it does not turn the trusted-local development setup into a production security platform.

Do not put secrets in chat. Inputs go to the configured model provider. Studio and
checkpoints expose private draft state; public schemas containing only messages do
not remove that data. The `.env.example` template disables tracing; existing `.env`
values are not changed. Enabled tracing can also store inputs/responses in LangSmith.
Filtering model copies does not sanitize stored chats or hide checkpoint state.
Keep the local development server bound to localhost while testing.

The recording service checks real code-recorded human approval, not model-generated
authority. Pending/refused/elapsed requests cannot be recorded as approved. Required entry:
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

The earlier Stage 4 offline run completed 339 tests (335 passed, four live checks
skipped); its separate two real-provider full-pipeline smoke checks also passed.
The **2026-11-02 priced-inventory regression completed 359 tests: 354 passed, five
opt-in live checks skipped**, including the budget-filtered real HTTP/stdio workflow.
The five opt-in live component checks were run separately and passed, including
grounded budget extraction. All 12 deterministic budget evaluation cases passed;
two live answer runs passed 12/13 cases. Retrieval results are mixed, with the
remaining exact-ID and challenge-paraphrase limits documented in RAG_EVALUATION.md.
Ruff lint/format and ty checks passed. No new package or provider change was made;
the baseline requirements, real `.env`, and populated project stores were unchanged.
See STAGE4_VALIDATION.md for the historical load evidence and its limitations.

**2026-11-02 conversation fixes:** the final offline run completed **389 tests in
329.240 seconds: 382 passed, seven opt-in live checks skipped**. It includes 27 new
fast screenshot/edge-case regressions and a real HTTP/stdio approval/refusal check
without a mandatory `/options` turn. The seven live component checks passed separately,
including raw parsed space-reference and negative/vague-correction extraction. Two
full real-provider screenshot conversations (approval/refusal) passed twice, retaining
initial choices through correction/confirmation; approved recording used one attempt
and refusal wrote no file. All 13 integration scenarios passed. A small six-conversation
local load regression at concurrency five and refreshed ten-way contention check passed;
these do not replace the historical 90-conversation benchmark or establish provider capacity.
Ruff lint/format, ty and whitespace checks passed. Tests used disposable stores, without
changing real `.env`, populated project data, the dependency lockfile, or baseline requirements.
Current demo screenshots in `presentation/` are diagnostic bug evidence; replace them
after a fresh Studio test before using them as successful presentation evidence.

Run the synthetic privacy evaluation with `.venv/bin/python -m guardrail_evaluation`.
It prints case IDs/counts, coverage metrics, and warm detector latency; no files are
written.

Run the RAG evaluation (see [RAG_EVALUATION.md](RAG_EVALUATION.md)); the live CLI needs
the VPN and uses temporary stores only:

```bash
uv run --no-sync --env-file .env python -m rag_evaluation --part retrieval  # or --part all
```

Neither Stage 1 evaluation is a load test. The completed full-system local load
and live smoke report is in [STAGE4_VALIDATION.md](STAGE4_VALIDATION.md).

Coverage includes Chroma/SQLite CRUD and persistence, validation/isolation/concurrency,
startup seeding/static reuse, SQLite reset/failure propagation, real Starlette lifespan,
eligibility-constrained semantic ranking/query privacy/thread isolation, context refresh/budgets/message preservation,
sync/async execution, and draft
collection/confirmation/duration handling, submission-time UTC periods, delayed review,
expiry boundaries, replay stability, public PII rejection, extraction redaction,
old-draft filtering, local-model readiness, guarded sync/async streaming, request/
location-term detection, "other options" exclusion, and RAG evaluation metrics/rules.
It does not establish complete privacy enforcement; Stage 4 load evidence and its
limits are documented separately.

Source checks:

```bash
uv tool run ty check
uv tool run ruff check app.py parking_agent.py pricing.py administrator_agent.py recording.py recording_client.py recording_server.py system_validation.py guardrails.py guardrail_evaluation.py rag_evaluation.py reservation_collection.py context_augmentation.py commons storage tests
uv tool run ruff format --check app.py parking_agent.py pricing.py administrator_agent.py recording.py recording_client.py recording_server.py system_validation.py guardrails.py guardrail_evaluation.py rag_evaluation.py reservation_collection.py context_augmentation.py commons storage tests
```

## Stage 4 validation

Run the isolated local system scenarios and load profiles:

```bash
PARKING_LIVE_TEST=0 LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
  .venv/bin/python -m system_validation --part offline --sessions 30 --concurrency 1 5 10
```

Opt-in real Azure/DIAL pipeline smoke (temporary stores, synthetic details, no tracing;
requires the existing credentials and VPN):

```bash
LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false PARKING_LIVE_TEST=1 \
  uv run --no-sync --env-file .env python -m system_validation --part live
```

The CLI prints safe JSON evidence and returns nonzero on failures. It starts an
owned localhost Agent Server with isolated Chroma/SQLite/output/checkpoints and
actual stdio MCP. Offline models are stateless fixtures, not measured provider
performance. Scripted decision-only resumes exercise the existing interrupt, not
a new administrator channel. Load/live checks stay explicit rather than running
as part of ordinary unit-test discovery. See [STAGE4_VALIDATION.md](STAGE4_VALIDATION.md)
for exact methods, timings, failure/retry/restart scenarios, limits and the narrow
lazy-storage-export fix that avoids loading Chroma/provider clients in the recorder.

## Selected deployment and handoff

Deployment for this assignment is the **Linux/local POSIX LangGraph development
server plus LangSmith Studio**, not a new cloud/container/production platform:

1. Clone the repository, install the existing lockfile, and configure `.env` as above.
2. Keep credentials/private data local and tracing disabled; connect the required VPN.
3. Run one server against each disposable SQLite/output location. Wait for startup
   readiness before using Studio; missing/empty Chroma requires initial embeddings.
4. Use the Studio collection → confirmation/automatic option recheck → explicit submission →
   Graph-mode Decision/Resume flow. Inspect the approved file only after a recorded-success reply.
5. Stop with Ctrl+C. A new lifespan deliberately resets SQLite and generated output;
   preserve nothing valuable in the dedicated recording directory. Chroma is reused.

For reproducing concurrent server tests, worker jobs are set explicitly to 10 in
the harness. The installed CLI's help advertises 10, but its underlying runtime's
omitted-argument fallback is 1; do not infer concurrent execution from client count.
The harness verifies actual overlapping node executions. Keep the development
server on localhost; Studio/state/debug access is trusted, not separate administrator
authentication. Do not share the same SQLite between server processes.

### Assignment repository links

After explicitly reviewing, committing and publishing the current changes, use this
same project repository in each stage's answer field; four separate repositories
are not required by the assignment:

| Stage | Repository |
| --- | --- |
| 1 — RAG/chatbot | https://github.com/narekhovhannisyanep/parking-space-reservation-agent |
| 2 — Human approval | https://github.com/narekhovhannisyanep/parking-space-reservation-agent |
| 3 — Approved MCP recording | https://github.com/narekhovhannisyanep/parking-space-reservation-agent |
| 4 — Full workflow/testing | https://github.com/narekhovhannisyanep/parking-space-reservation-agent |

The URL comes from local Git remote configuration. Remote contents/accessibility
and assignment answer fields were not verified or changed during this validation.
No commit or push is automatic. Presentation and CI/CD/infrastructure remain optional
extra credit, not missing core features. Optional retrieval tuning requires a separate
decision and re-measurement; no production/scaling backlog is implied.

## References

- [LangGraph local server](https://docs.langchain.com/langsmith/local-server)
- [Server startup lifespan](https://docs.langchain.com/langsmith/custom-lifespan)
- [Studio setup](https://docs.langchain.com/langsmith/quick-start-studio)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph interrupts and typed Studio responses](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Using Studio](https://docs.langchain.com/langsmith/use-studio)
- [LangChain Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- [Python SQLite](https://docs.python.org/3/library/sqlite3.html)
- [Official MCP Python SDK](https://py.sdk.modelcontextprotocol.io/)
- [MCP stdio transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP local-server security](https://modelcontextprotocol.io/specification/2026-07-28/basic/security_best_practices#local-mcp-server-compromise)
