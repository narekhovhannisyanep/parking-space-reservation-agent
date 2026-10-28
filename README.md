# parking-space-reservation-agent
The goal of the project is to develop an intelligent chatbot that can interact with users, provide information about parking spaces, handle the reservation process, and involve a human administrator for confirmation.

## Run with LangGraph CLI and LangSmith Studio

Use Python 3.12 or newer and install the locked dependencies, including the CLI,
with `uv sync --dev`.
Create a project-root `.env` using `.env.example` as a template, then set
`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `OPENAI_API_VERSION` for your
Azure-compatible endpoint. `AZURE_OPENAI_DEPLOYMENT` optionally overrides the
existing DIAL deployment in `commons/constants.py`.
Set `LANGSMITH_API_KEY` for your LangSmith account and sign in to LangSmith in your
browser. The CLI loads `.env` via `langgraph.json`; application code does not load it.
Start the development server from the project root:

```bash
.venv/bin/langgraph dev
```

Alternatively, run `uv run langgraph dev`. The server supports hot reload and opens
LangSmith Studio. If it does not open automatically, use the Studio URL printed by
the CLI, normally:

`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

In Studio, select **parking_space_reservation**, switch to **Chat** mode, and send
a message. Continue in the same thread to preserve conversation history, or create
a new thread for an independent conversation. **Graph** mode shows nodes, state,
and execution details. The local API documentation is at `http://127.0.0.1:2024/docs`.
Use Ctrl+C in the server terminal to stop it. `python app.py` is no longer a chat UI.

The development server manages checkpoints and saves local state under
`.langgraph_api/` (git-ignored). This is development storage, not a production
reservation database. Keep the server bound to localhost; production requires
deployment, authentication, authorization, and an appropriate customer-facing UI.
Studio is suitable for development, demos, and internal testing, not an isolated
public customer interface: it provides access to thread state and debugging tools.

## Agent structure

- The graph starts with `reservation_collection`. Information turns continue through
  `context_augmentation → parking_assitant → END`; deterministic collection/confirmation
  replies go directly to `parking_assitant → END`, without retrieval or another model call.
  `parking_assitant` retains its user-selected spelling and sync/native async implementations.
- `app.py` is a thin entry point: one `ParkingSpaceReservationAgent` instance exports
  its compiled graph as `agent` for `langgraph.json`.
- `parking_agent.py` owns `AgentConfig`, model initialization, and graph construction.
  `AgentConfig` is an immutable, non-secret configuration shared by the instance's
  nodes. The default Azure/DIAL client is initialized on first use and reused;
  concurrent initialization is protected. Model calls use native async execution
  in server runs, with a 60-second timeout and up to two automatic retries.
- Conversation and per-request reservation state belong in LangGraph state,
  scoped to a thread, **not** mutable attributes shared by all users on the class.
  `ParkingState` is a dataclass with a `messages` field and the `add_messages` reducer,
  retaining the same messages-based API/Studio schema and update-by-ID behavior.
  This avoids the `MessagesState` TypedDict protocol-bound incompatibility in `ty`.
  Internal `retrieval_context`, `reservation`, and `reservation_reply` are not public
  input/output fields. `reservation` holds JSON-compatible private draft/request state,
  validated at node boundaries, not custom Python checkpoint objects. Input/output schemas
  expose only messages; checkpoints/Studio/streams still expose internal state and personal data.
  The class owns resources and workflow code; it is not a third orchestrator LLM.
- `storage/` supplies class-owned `static_store` (persistent Chroma via
  `langchain-chroma`) and `dynamic_store` (Python's built-in SQLite). `AgentConfig.storage`
  holds immutable `StorageConfig`; both helpers initialize their databases on first use.
  They are trusted Python APIs, not model tools or shared conversation state.

Importing `app` creates the instance and graph but does not read `.env`, initialize
an API client, create databases, or make network calls. Tests can inject a model
through `ParkingSpaceReservationAgent(llm_client=...)` and an optional checkpointer
for standalone use. Embeddings and either storage helper can also be injected through
the `embeddings`, `static_store`, and `dynamic_store` constructor arguments.
The server export intentionally has no custom checkpointer:
Agent Server supplies persistence. Without the server or an injected checkpointer,
information-only direct callers must supply conversation history themselves. Multi-turn
draft collection/confirmation requires Agent Server or an injected checkpointer; messages
alone cannot reconstruct the private draft from the public input schema.

The four stages in `AGENTS.md` are the current scope: RAG/chat and required booking
details, human administrator approval through a second agent, approved file recording,
and LangGraph orchestration with privacy, evaluation, and integration/load testing.
Real-world vehicle-fit and facility-management features are out of scope.

This prototype has graph-wired retrieval and structured reservation-detail collection.
Live availability enforcement, submission, administrator approval, allocation, and recording
are **not implemented**.
It cannot complete reservations.
Inputs are sent to the configured model provider. Keep secrets out of chat and leave
LangSmith tracing disabled unless personal-data handling and access controls are configured.
The template disables tracing; an existing `.env` is not changed automatically.
Studio can still connect with tracing disabled. This does not prevent sending inputs
to your model provider or saving them in local checkpoints. If you enable tracing,
prompts, responses, and personal data may also be stored in LangSmith. Sanitized
graph errors do not replace access controls, checkpoint protection, or trace redaction.
Expected OpenAI API, client-configuration, and storage I/O/SQLite errors are sanitized; programming
errors propagate so they can be diagnosed rather than silently masked.

## Reservation draft collection

`reservation_collection.py` defines strict, extra-field-rejecting models for proposed updates,
the private draft, the collection session, and `ValidatedReservationRequest`. The latter is
**collected information, not a submitted or approved reservation**. Session state is isolated
by graph thread; no draft fields live on the shared agent instance or in either public store.

1. A lazy structured-output wrapper uses the existing model to propose reservation intent and
   field patches. Only the latest human text and current draft are supplied, not retrieved
   evidence or assistant-generated facts. Information-only proposals cannot contain field updates.
2. Python accepts only allowlisted fields whose proposed values occur verbatim in the latest
   human text. It trims/bounds single-line values and rejects control characters and `|`.
   Unmentioned fields are preserved; missing details are never filled with defaults. This
   substring check prevents invented values, but cannot prove correct semantic field assignment;
   users must review the summary. Model extraction accuracy still needs evaluation.
3. Collect first name, surname, `car_number` (any user-provided vehicle identifier), and full
   start/end dates and times. Current collection accepts `YYYY-MM-DD HH:MM` or ISO `T` timestamps,
   optionally including seconds and explicit offsets. Date-only, shorthand, relative, and
   ambiguous periods are clarified rather than interpreted by the model. No vehicle type,
   measurements, or plate format is required.
4. Resolve an explicit IANA timezone (such as `Asia/Yerevan`), `UTC`, or `UTC+04:00`/`+04:00`.
   Start/end with explicit offsets need no separate timezone field. Ambiguous abbreviations,
   nonexistent/ambiguous clock-change times, and timezone/offset conflicts are rejected;
   supply explicit offsets to disambiguate daylight-saving transitions. IANA zones require
   system timezone data; UTC/fixed offsets remain usable without it. Normalize validated
   instants to UTC, require a future start and `end > start`, and revalidate before confirmation.
5. Code generates a draft ID, revision, and exact confirmation summary. A later user reply
   confirms the displayed revision. Corrections, clearing fields, rejected/unreadable updates,
   and trusted location/timezone-data changes invalidate prior confirmation. A readiness
   acknowledgment never claims submission, administrator approval, allocation, or recording.

Example first message in Studio: “Reserve parking. First name Demo, surname Tester, car
number demo-bicycle-1. Start 2030-06-01 10:00, end 2030-06-01 12:00, timezone UTC+04:00.”
Use your actual future dates; this is synthetic example data.

| Reply/command | Effect |
| --- | --- |
| `/confirm N` | Confirm the exact displayed draft revision N |
| `yes`, `yes.`, `confirm`, `I confirm` | Confirm only immediately after the matching summary |
| `no`, `no.` | Decline details confirmation; provide corrections or cancel |
| A correction in natural language | Propose field changes; show a new summary after validation |
| `/clear FIELD` | Remove one of `first_name`, `surname`, `car_number`, `start`, `end`, or `timezone` |
| `/cancel` | Clear the current draft and confirmation, without deleting chat/checkpoint history |
| `/new` | Start a new empty draft with a new code-generated ID |

Collection statuses are `idle`, `collecting`, `awaiting_confirmation`, `ready`, and `cancelled`.
`ready` means user-confirmed details for the **future** approval workflow. No reservation ledger,
capacity hold, notification, approval decision, MCP call, or file entry is created. Future
submission/allocation code must independently revalidate the request and trusted human decision.

General information questions still use RAG, including while a draft is incomplete. Such turns
make one extraction call plus the ordinary answer call; collection turns use one extraction call
and a deterministic reply; standalone commands need neither a model call nor storage access.
Extraction parsing/provider failures return a sanitized message and clear prior confirmation;
unexpected programming errors propagate. Raw structured responses are not added to chat or stores,
but callbacks/traces may still capture them. Protect provider inputs, checkpoints, streams,
retention, and access controls before using real customer details.

### Model compatibility

`PARKING_EXTRACTION_METHOD` defaults to `function_calling`; `json_schema` is an explicit alternative.
Both require actual support from the deployed Azure/DIAL model and endpoint. There is no silent
fallback to prose parsing, and no assumption that Azure API compatibility guarantees support.
The wrapper is initialized once per instance; tests can inject `reservation_extractor` and an
aware `clock`. No new dependency or `.env` change is required.

Offline tests do not prove deployment compatibility. An opt-in smoke suite uses synthetic data,
mocked public stores, a 15-second timeout, and no retries:

```bash
LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false PARKING_LIVE_TEST=1 \
  uv run --no-sync --env-file .env python -m unittest discover -s tests -p test_reservation_model_live.py -v
```

These tests are skipped in normal offline runs. The implementation-time endpoint check timed
out with `OpenAITimeoutError`; live compatibility and model accuracy remain unverified. Check
endpoint connectivity/credentials, then rerun; change the extraction method only if the actual
deployment supports it. The existing provider, `.env`, and databases were left unchanged.

## Persistent parking data

Both stores are local and persistent across process restarts. Static demo data is
available through an explicit seed command; dynamic data remains unseeded. Starting
Studio does not initialize storage. A nonblank information turn reads both stores and
may create empty databases, but never seeds them. Writes require explicit trusted Python calls.

### Per-turn context augmentation

`context_augmentation.py` implements deterministic retrieval before information-answer model calls:

1. Inspect up to 2,000 characters of the latest human text (not assistant/tool output
   or full history). Build the embedding query only from fixed public-topic terms:
   entrance, lift, location/directions, hours, prices, and availability. Unrecognized
   text uses `parking information`; raw names, vehicle identifiers, dates, credentials,
   and arbitrary user text are never forwarded as embedding-query values. Blank/nontext
   messages skip retrieval. Deterministic collection replies bypass both stores.
2. Retrieve up to four static spaces within the trusted deployment location, then
   read that location's `working_hours`, `prices`, and `availability` SQLite snapshots.
3. Build JSON evidence with space IDs/sources and intact dynamic values/write timestamps.
   Dynamic facts are the only operational-fact source: model instructions give them
   precedence over conflicting static prose or earlier chat. Missing/null values mean
   unknown, never free/open/available. Static documents and databases are not rewritten.
4. Append evidence to a **copy** of the latest human message for the model. Preserve
   its ID/metadata/multimodal blocks and leave stored conversation messages unchanged.

Evidence is refreshed on information turns and cleared on collection turns, not accumulated
in chat history. `ContextConfig`,
available as `AgentConfig.context`, defaults to `location_id="demo-central"`,
`retrieval_k=4`, `max_query_chars=2000`, and `max_context_chars=12000`. The latter
caps the complete added evidence in characters, not tokens or total conversation size.
Whole dynamic snapshots get budget priority; oversized records are omitted with
explicit status/counts rather than misleading partial values. All three public dynamic
fact kinds are supplied because intent-specific schemas/routing remain undecided.
`PARKING_LOCATION_ID` overrides the deployment scope; user text does not change it.
Location scoping is not authorization. Follow-up query rewriting and exact-ID extraction
are not implemented; model history remains available for conversation understanding.

Async runs offload synchronous storage/embedding operations to a worker thread and
retain native async model calls. Retrieval failures stop the model call instead of
silently answering without evidence. Empty stores return explicit missing information
without initializing an embedding client. Populated stores require embedding credentials
and send the fixed-term public query to the embedding provider. This allowlist is data minimization,
not a general PII detector. It deliberately narrows free-form semantic retrieval and does not
forward user-supplied exact space IDs. Follow-up rewriting, expanded public-intent handling,
and retrieval-accuracy/latency evaluation remain needed; never restore raw query forwarding
without an appropriate privacy boundary.

Precedence and evidence-only instructions guide the model; they do not enforce output
grounding or prevent prompt injection. Snapshots have no freshness/period guarantee.
Availability payloads remain schema-free public observations: this change does not
interpret them as per-space allocation rules, filter unavailable candidates, or promise
booking availability. Top-K retrieval is not an exhaustive inventory/count.

### Static space inventory: Chroma

The exported instance's `app.parking_agent.static_store` exposes `upsert_documents(documents)`,
`search(query, location_id=..., k=4, filters=SpaceFilters(...))`, and `delete_documents(ids)`.
`ParkingSpace` in `storage/spaces.py` validates structured inventory and generates
LangChain documents with canonical `{location_id}:space:{space_id}` IDs.

- Identity: `space_id`, `location_id`, `label`.
- Search/context/provenance: concise public `description` and reviewed `source`.

Every space is assumed to fit any vehicle for this project. There are no dimensions,
vehicle categories, or accessibility/charging/covered flags. Directions stay in the
description; no separate level/zone fields are needed. This assumption simplifies
the prototype; it is not a physical-fit guarantee for a real parking facility.
The required `car_number` means a user-provided vehicle identifier even for vehicles
without plates; booking does not require vehicle type, measurements, or plate format.

Generated Chroma metadata has `record_type=parking_space` and integer `schema_version=2`.
`SpaceFilters` allows only exact `space_id` lookup. Filters run **before** semantic
ranking and always retain location/type/schema scope.
This is static candidate retrieval, **not** live availability or allocation enforcement.
See [PARKING_DATA_MODEL.md](PARKING_DATA_MODEL.md) for the minimal field contract and
future booking responsibilities.

Repeated upserts replace documents with the same ID; IDs are unique across the whole
collection, so include location in IDs when appropriate. Searches filter by explicit
location and return an empty list when no matching documents exist, without calling
the embedding provider. Empty upsert/delete batches are no-ops.

The default embedding client uses `AzureOpenAIEmbeddings`, the existing Azure/DIAL
endpoint, API key, and API version, and the deployment/model constants in
`commons/constants.py`. It initializes only on the first actual embedding request.
Text is sent directly without tokenizer/model downloads; future ingestion must chunk
documents to the configured embedding model's input limits. Changing the embedding
model requires reindexing into a new collection, even if vector dimensions match.
Expected embedding configuration/API errors are sanitized.

### Seed the mock space inventory

`storage/parking_spaces.json` contains **eight individually identified mock spaces**
for the fictional `demo-central` location, with source `mock-space-inventory-v2`.
They differ only by identity and public directions, not by which vehicles can use them.
There are no invented tram/toilet guides or workflow instructions in the corpus,
and no prices, working hours, customer details, availability flags, holds, or reservations.
Individual-space allocation is the chosen model; production location count remains open.

The smaller corpus demonstrates space retrieval and exact filtering, not complete
application functionality or semantic accuracy. Per-turn token usage depends on
selected context, not the total number of indexed spaces. Avoid merging independent
spaces into one large document; retain individually filterable identities. Evaluate
with a separate question/expected-space dataset, including exact IDs, missing spaces,
paraphrases, and unavailable candidates once dynamic wiring exists. The fixture alone
does not cover all required general information, location, hours, or price questions.

Validate offline first, from the project root:

```bash
.venv/bin/python -m storage.seed --dry-run
```

Load the existing `.env` explicitly and embed/upsert the spaces into the configured
Chroma collection (defaults to `parking_spaces_v2` under `.data/chroma`):

```bash
uv run --env-file .env python -m storage.seed
```

This calls the configured embedding provider and requires the Azure/DIAL settings
described above. Application imports and the seed module do not load `.env` themselves.
The command validates the entire file before initializing storage. Repeating it updates
the same stable `demo-central:space:*` IDs rather than creating duplicates; unrelated IDs are
preserved, and SQLite is untouched. It does not delete previously seeded records omitted
from a newer file; delete obsolete IDs explicitly through `delete_documents`.

For retrieval, pass `location_id="demo-central"` and optional validated `SpaceFilters`
to `static_store.search(...)`.
The chat graph retrieves automatically within `AgentConfig.context.location_id`.

To expand the dataset, append reviewed structured spaces using the exact inventory
fields listed in [PARKING_DATA_MODEL.md](PARKING_DATA_MODEL.md); do not hand-author a
second `page_content`/metadata copy. There is no fixed record-count limit. You can
also supply a different reviewed inventory file:

```bash
.venv/bin/python -m storage.seed --file path/to/reviewed-spaces.json --dry-run
uv run --env-file .env python -m storage.seed --file path/to/reviewed-spaces.json
```

The bundled file resolves relative to the seed module, independent of the working
directory; an explicit `--file` path resolves relative to your current working directory.
Keep each document within the embedding model's input limits; automatic chunking is
not implemented. Review custom text for sensitive information and source authenticity:
schema validation alone does not enforce either. Keep demo data out of production.

#### Migration from previous corpora

The old guide and dimension/vehicle-specific schemas are no longer accepted.
The default collection is now `parking_spaces_v2`, leaving previously persisted
`parking_static` and `parking_spaces_v1` data untouched. If your existing `.env`
sets an older collection, change `PARKING_CHROMA_COLLECTION` to `parking_spaces_v2`
and run the seed command. Use a fresh collection: Chroma merges metadata, so
upserting over old-schema IDs in an old collection can retain removed restrictions.
Search filters exclude untouched old-schema records, but are not a metadata migration.
Existing `.env` and project databases are not changed by this refactor.

Retire the unused
collection through a reviewed maintenance operation after backup; never delete a
live database directory to migrate. Updating the JSON does not delete previously
indexed spaces omitted from it. Removal/update approval and ingestion synchronization
are separate future work; the seed CLI itself does not enforce human approval.

### Dynamic information: SQLite

`app.parking_agent.dynamic_store` exposes `upsert_fact(location_id, kind, value)`,
`get_fact(location_id, kind)`, `list_facts(location_id)`, and
`delete_fact(location_id, kind)`. Allowed kinds are `working_hours`, `prices`, and
`availability`. `parking_dynamic_facts` has a `(location_id, kind)` primary key,
a JSON value, and a UTC `updated_at` write timestamp. Missing facts return `None`
or an empty list, never defaults such as zero available spaces. Writes are atomic
upserts with parameter-bound SQL; connections are committed/rolled back and closed
per operation, rather than shared across graph workers.

These existing JSON snapshots remain public operational observations, not private
reservation storage. See [PARKING_DATA_MODEL.md](PARKING_DATA_MODEL.md).
Payloads must be JSON-compatible, with string object
keys and finite numbers. `updated_at` is not a freshness guarantee; refresh rules
and upstream observation times remain future work. An availability snapshot is
**not** an allocation, hold, or reservation ledger. Availability for a requested period
must eventually account for overlapping confirmed reservations and any chosen pending
holds. Public operational snapshots must never contain private request data.

### Configuration and lifecycle

Optional environment settings are documented in `.env.example`:

| Setting | Default |
| --- | --- |
| `PARKING_CHROMA_DIRECTORY` | `.data/chroma` |
| `PARKING_CHROMA_COLLECTION` | `parking_spaces_v2` |
| `PARKING_SQLITE_PATH` | `.data/parking.sqlite3` |
| `PARKING_LOCATION_ID` | `demo-central` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | `DIAL_TEXT_EMBEDDING_3_SMALL` |
| `AZURE_OPENAI_EMBEDDING_MODEL` | `TEXT_EMBEDDING_3_SMALL` |

Relative paths are anchored to the project root, not the shell's current directory.
The CLI loads `.env`; direct Python callers must load/provide their environment
themselves. Existing `.env` values are unchanged. Run `uv sync --dev` to install the
updated lockfile. SQLite needs no extra package; Chroma needs no standalone server.
Chroma persists automatically and its local client follows the owning process's
lifetime. Async context augmentation runs synchronous storage APIs off the event loop.
Back up the databases using database-supported backup
procedures; don't delete live files or copy a partially written directory.

Initialization, reviewed seeding/ingestion, readiness validation, and refresh are
separate operations. Startup bootstrap/readiness wiring is not implemented. Static
knowledge changes require administrator approval before trusted ingestion; reservation
approval does not modify Chroma. Pending holds belong in private transactional SQL
state and can change availability before approval, without confirming a booking.

`.data/` and database sidecar files are git-ignored. Chroma anonymized telemetry is
disabled. Metadata validation is **not** a PII detector or an authorization boundary:
trusted ingestion must review document text, sources, and JSON values. Never store
personal reservation details, chats, credentials, or approval decisions in these
public-fact stores. Location filtering is not tenant authorization. Protect local
files with OS permissions, and add authentication, safe logging, access controls,
and retention policies before production or customer-facing deployment.

## System prompt

The user-facing prompt is defined in `commons/constants.py`. The approved architecture uses
two agents coordinated by LangGraph/Python code. Allocation, approval enforcement, reservation
state, and file formatting/writing are programmatic responsibilities, not LLM decisions.
The current prototype implements draft collection/validation/confirmation only, not those services.

## Offline checks

With the project dependencies installed in `.venv`, run:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

These checks cover the CLI export, shared configuration/client lifecycle, sync and
async graph execution, thread isolation, sanitized errors, and the prompt contract.
Storage checks use temporary directories and offline deterministic embeddings to test
Chroma/SQLite CRUD, new-process persistence, location/collection isolation, lazy
initialization, input validation, concurrent workers, and transaction cleanup.
Inventory/seed checks also cover the eight-space fixture, strict fields/identity,
minimal vehicle-agnostic fields, exact space lookup, old-schema exclusion, extensibility,
dry-run behavior, invalid batches, repeatable ingestion, and unrelated-document preservation.
Context checks cover real Chroma/SQLite graph integration, dynamic precedence instructions,
snapshot refresh/deletion, missing values, budgets, message preservation, sync/async parity,
event-loop offloading, and thread/location isolation.
Collection checks cover strict schemas, verbatim field grounding, required details, multi-turn
updates, UTC normalization, clock-change ambiguity, future/ordered periods, confirmation revisions,
corrections/clearing/cancellation, fail-closed extraction, information-turn continuity, native async
execution, checkpoint round trips, thread isolation, public input filtering, and query privacy.
They do not seed project data, make API calls, or evaluate live model behavior,
semantic retrieval accuracy, or production security enforcement.

Type, source lint, and formatting checks can run through uv's temporary tool environments
without changing the project dependencies:

```bash
uv tool run ty check
uv tool run ruff check app.py parking_agent.py reservation_collection.py context_augmentation.py commons storage tests
uv tool run ruff format --check app.py parking_agent.py reservation_collection.py context_augmentation.py commons storage tests
```

Ruff enables `BLE001` to prevent broad exception catches from returning.

References: [local server](https://docs.langchain.com/langsmith/local-server),
[Studio setup](https://docs.langchain.com/langsmith/quick-start-studio), and
[server-managed persistence](https://docs.langchain.com/oss/python/langgraph/persistence),
[Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma),
[persistent Chroma clients](https://docs.trychroma.com/docs/run-chroma/clients),
[Azure embeddings](https://reference.langchain.com/python/langchain-openai/embeddings/azure/AzureOpenAIEmbeddings),
and [SQLite](https://docs.python.org/3/library/sqlite3.html),
[structured model output](https://docs.langchain.com/oss/python/langchain/models#structured-output),
[Azure structured-output methods](https://reference.langchain.com/python/langchain-openai/chat_models/azure/AzureChatOpenAI/with_structured_output),
and [IANA timezone handling](https://docs.python.org/3/library/zoneinfo.html).
