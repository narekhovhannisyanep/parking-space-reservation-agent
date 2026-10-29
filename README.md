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

## Current status

| Stage | Implemented | Still needed |
| --- | --- | --- |
| 1: RAG/chatbot | Persistent Chroma/SQLite stores, graph-wired retrieval, draft collection/validation/user confirmation | Complete parking-information data, required guardrails/filtering, live accuracy checks, performance evaluation/report |
| 2: Human approval | Two-agent approach selected | Administrator agent, communication channel, request escalation, approval/refusal handling |
| 3: Approved recording | File format and Python-owned recording responsibility selected | MCP service (or permitted fallback), access protection, approved-only file writes |
| 4: Orchestration | Conversation graph and offline integration tests | Complete approval/recording pipeline, end-to-end/load tests, completed usage/deployment documentation |

**The current application cannot complete reservations.** User-confirmed details
are not administrator approval. Submission, allocation, administrator communication,
MCP calls, and approved-reservation recording are not implemented.

## Setup and run

Use Python 3.12 or newer. Install the locked dependencies:

```bash
uv sync --dev
```

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
initialize clients/databases, or make API calls. Clients and stores initialize lazily.
Agent Server manages thread checkpoints under `.langgraph_api/` (git-ignored).

## Project structure and current workflow

| File/directory | Responsibility |
| --- | --- |
| `app.py`, `langgraph.json` | CLI graph export/configuration |
| `parking_agent.py` | Agent-owned configuration/resources and deterministic graph construction |
| `commons/constants.py` | Current model constants and user-facing system prompt |
| `context_augmentation.py` | Public-information retrieval and model-only evidence augmentation |
| `reservation_collection.py` | Draft extraction, validation, user confirmation, and cancellation |
| `storage/` | Chroma static inventory, SQLite dynamic facts, configuration, and explicit seeding |
| `tests/` | Offline storage, graph, collection, and opt-in live-model checks |

Current graph:

```text
START → reservation_collection
  information turn → context_augmentation → parking_assitant → END
  collection reply → parking_assitant → END
```

`parking_assitant` retains the user-selected spelling. The accepted complete
architecture uses two agents coordinated by Python/LangGraph, not a third orchestrator
LLM. The human administrator alone approves; Python owns allocation, reservation
state, approval enforcement, and file formatting/writing.

The agent owns shared resources, while conversation/request state belongs to graph
threads. Tests can inject models, extractors, clocks, embeddings, stores, and a
checkpointer. Direct callers outside Agent Server need a checkpointer for multi-turn
draft state; the public messages alone do not reconstruct it.

## Persistent parking information

The selected static/dynamic split is an optional assignment improvement we are
keeping: local persistent Chroma for static information, built-in SQLite for hours,
prices, and availability. Neither requires a separate database server.

Both databases initialize on first storage use, not server startup. Information
turns may create empty databases but never seed them. Deterministic collection
replies skip storage. Personal reservation details and chats are not ingested into
either public-information store.

### Static inventory and seeding

`StaticParkingStore` exposes `upsert_documents`, `search`, and `delete_documents`.
`ParkingSpace` validates individual inventory records and generates LangChain
documents. `SpaceFilters` supports exact space-ID filtering within mandatory
location/type/schema scope before semantic ranking.

`storage/parking_spaces.json` contains eight mock spaces for `demo-central`, differing
by identity and public directions. Every space fits any vehicle by project assumption;
no dimensions, vehicle categories, or facility-feature matching are modeled.
The fixture is not complete data for all required general parking/location questions.

Validate the fixture offline:

```bash
.venv/bin/python -m storage.seed --dry-run
```

Explicitly load `.env` and embed/upsert the spaces:

```bash
uv run --env-file .env python -m storage.seed
```

This calls the configured embedding provider. To use another reviewed inventory:

```bash
.venv/bin/python -m storage.seed --file path/to/reviewed-spaces.json --dry-run
uv run --env-file .env python -m storage.seed --file path/to/reviewed-spaces.json
```

Repeated upserts update the same IDs; unrelated IDs are preserved. Omitted records
are not deleted automatically. SQLite is untouched. Static-data changes require
administrator approval as an accepted user addition, separate from reservation
approval; the current seed CLI does not enforce it.

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

Dynamic data is unseeded. Missing facts return `None` or an empty list, not invented
defaults. Payloads are JSON-compatible but have no domain-specific schemas yet.
These are public observations, not a reservation ledger, capacity holds, or
availability checks for a requested period. A write timestamp does not prove freshness.

### Information retrieval

For information turns, code builds queries from fixed English public-topic terms
(entrance, lift, location/directions, hours, prices, availability), never raw user
text or draft values. Unknown topics use `parking information`. It retrieves static
spaces and reads dynamic facts within the configured location.

JSON evidence is appended only to a copy of the latest user message for the model.
Dynamic snapshots have evidence-budget priority and are the instructed source for
operational facts. Missing/null/omitted values mean unknown. Top-K retrieval is not
exhaustive inventory or allocation. Async storage runs off the event loop.

Current defaults: location `demo-central`, K=4, inspected user text=2,000 characters,
evidence=12,000 characters. `ContextConfig` holds these settings. Fixed queries narrow
free-form retrieval; follow-up rewriting and user-text exact-ID extraction are not
implemented. Model instructions are not enforced output grounding or complete privacy guards.

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
identifier, not necessarily a plate), and start/end dates/times. Current code accepts
full `YYYY-MM-DD HH:MM` or ISO `T` timestamps, optionally seconds/UTC offsets, with
an explicit IANA timezone or UTC offset when needed. It normalizes to UTC, requires
a future start and end after start, and asks for clarification on relative/shorthand
dates, timezone conflicts, and ambiguous/nonexistent daylight-saving times.

Example synthetic message (use actual future dates):

“Reserve parking. First name Demo, surname Tester, car number demo-bicycle-1.
Start 2030-06-01 10:00, end 2030-06-01 12:00, timezone UTC+04:00.”

Python validates proposed verbatim field values and renders a revisioned summary.
Substring checks do not prove correct semantic extraction; the user reviews the summary.

| Reply/command | Effect |
| --- | --- |
| `/confirm N` | Confirm displayed revision N |
| `yes`, `yes.`, `confirm`, `I confirm` | Confirm only immediately after the matching summary |
| `no`, `no.` | Decline details confirmation |
| A natural-language correction | Propose changes and validate/show a new summary |
| `/clear FIELD` | Remove `first_name`, `surname`, `car_number`, `start`, `end`, or `timezone` |
| `/cancel` | Clear the draft/confirmation, not chat history |
| `/new` | Start a new empty draft |

States: `idle`, `collecting`, `awaiting_confirmation`, `ready`, `cancelled`.
Changes and rejected/unreadable updates invalidate confirmation. `ready` is collected,
user-confirmed information only. General information turns still use RAG while collecting.
Current commands/revisions/timezone rules describe implemented behavior, not additional
assignment requirements that cannot be reconsidered during refactoring.

### Model compatibility

`PARKING_EXTRACTION_METHOD` defaults to `function_calling`; `json_schema` is an explicit
alternative. Actual endpoint support is required; there is no prose-parsing fallback.
Offline tests do not prove live compatibility. The earlier synthetic endpoint check
timed out with `OpenAITimeoutError`; live compatibility/accuracy remain unverified.

Opt-in smoke test (synthetic data, mocked public stores, no tracing):

```bash
LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false PARKING_LIVE_TEST=1 \
  uv run --no-sync --env-file .env python -m unittest discover -s tests -p test_reservation_model_live.py -v
```

## Required data protection and recording safety

Stage 1 requires sensitive-data guardrails/filtering; Stage 3 requires a secure,
reliable recording service resistant to unauthorized access. They remain unfinished,
not optional, but do not imply building a production security platform.

Do not put secrets in chat. Inputs go to the configured model provider. Studio and
checkpoints expose private draft state; public schemas containing only messages do
not remove that data. The `.env.example` template disables tracing; existing `.env`
values are not changed. Enabled tracing can also store inputs/responses in LangSmith.
Metadata validation and location filtering are not sensitive-content detection.
Keep the local development server bound to localhost while testing.

The future recording service must receive real human approval, not model-generated
authority. Pending/refused requests must not be recorded as approved. Required entry:
`Name | Car Number | Reservation Period | Approval Time`. Python owns the format;
approval and successful writing are separate results.

## Verification

Run the offline suite (temporary databases and deterministic embeddings, no API calls;
live tests are skipped unless explicitly enabled):

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Coverage includes Chroma/SQLite CRUD and persistence, validation/isolation/concurrency,
graph context refresh/budgets/message preservation, sync/async execution, and draft
collection/confirmation/timezone handling. It does not establish live model accuracy,
complete privacy enforcement, or completion of the assignment's evaluation/load tests.

Source checks:

```bash
uv tool run ty check
uv tool run ruff check app.py parking_agent.py reservation_collection.py context_augmentation.py commons storage tests
uv tool run ruff format --check app.py parking_agent.py reservation_collection.py context_augmentation.py commons storage tests
```

## Next steps

Finish the four stages using the retained choices in AGENTS.md. Decide administrator
channel, MCP implementation/output location, and the simple allocation/approval flow
when working on those steps. Pending holds were discussed, not selected. Do not build
hold-management, scaling, or production infrastructure speculatively.

## References

- [LangGraph local server](https://docs.langchain.com/langsmith/local-server)
- [Studio setup](https://docs.langchain.com/langsmith/quick-start-studio)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangChain Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- [Python SQLite](https://docs.python.org/3/library/sqlite3.html)
