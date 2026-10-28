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

- The conversational graph is `START → parking_assitant → END`. `parking_assitant`
  is the user-facing model-call node (not a tool node), with sync and native async implementations.
- `app.py` is a thin entry point: one `ParkingSpaceReservationAgent` instance exports
  its compiled graph as `agent` for `langgraph.json`.
- `parking_agent.py` owns `AgentConfig`, model initialization, and graph construction.
  `AgentConfig` is an immutable, non-secret configuration shared by the instance's
  nodes. The default Azure/DIAL client is initialized on first use and reused;
  concurrent initialization is protected. Model calls use native async execution
  in server runs, with a 60-second timeout and up to two automatic retries.
- Conversation and future per-request reservation state belong in LangGraph state,
  scoped to a thread, **not** mutable attributes shared by all users on the class.
  `ParkingState` is a dataclass with a `messages` field and the `add_messages` reducer,
  retaining the same messages-based API/Studio schema and update-by-ID behavior.
  This avoids the `MessagesState` TypedDict protocol-bound incompatibility in `ty`.
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
direct callers must supply conversation history themselves.

This remains a conversational prototype with an explicitly seedable storage layer: graph-wired RAG,
live availability, administrator approval, allocation, and recording are **not implemented**.
It cannot complete reservations.
Inputs are sent to the configured model provider. Keep secrets out of chat and leave
LangSmith tracing disabled unless personal-data handling and access controls are configured.
The template disables tracing; an existing `.env` is not changed automatically.
Studio can still connect with tracing disabled. This does not prevent sending inputs
to your model provider or saving them in local checkpoints. If you enable tracing,
prompts, responses, and personal data may also be stored in LangSmith. Sanitized
graph errors do not replace access controls, checkpoint protection, or trace redaction.
Only expected OpenAI API and client-configuration errors are sanitized; programming
errors propagate so they can be diagnosed rather than silently masked.

## Persistent parking data

Both stores are local and persistent across process restarts. Static demo data is
available through an explicit seed command; dynamic data remains unseeded. Starting
Studio or sending a chat message does not seed or create either database: the graph
does not yet call these helpers. Writes require explicit trusted Python calls.

### Static information: Chroma

The exported instance's `app.parking_agent.static_store` exposes `upsert_documents(documents)`,
`search(query, location_id=..., k=4)`, and `delete_documents(ids)`.
Ingestion accepts LangChain `Document` objects with a stable, nonblank `id`, nonblank
text, and exactly these nonblank string metadata fields:

- `location_id`: an explicit parking-location identifier.
- `source`: the reviewed public information source.
- `topic`: `general`, `location`, or `booking_process`.

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

### Seed the static examples

`storage/static_examples.json` contains **30 example records** for one fictional
location, `demo-central`, with source `demo-parking-guide-v1`:

- 10 general records: overview, vehicle types, height clearance, accessibility,
  charging facilities, static/dynamic fact boundaries, speed limits, bay-use rules,
  belongings, and visitor toilets.
- 9 location records: address, vehicle entrance/exit, pedestrian access,
  public transport, wayfinding, visitor assistance, lift location, and motorcycle parking.
- 11 booking-process records: required details, incomplete requests, pending approval,
  refusal, explicit human approval, recording outcomes, availability changing during
  approval, period clarification, corrected details, privacy, and payment-versus-approval.

Records read as normal parking-guide entries, without repeated demo disclaimers.
Booking-state records describe the intended workflow,
not real requests, implemented services, or authoritative approval decisions. No personal
reservation data, live prices, working hours, or availability snapshots are seeded.
One demo location does not decide the eventual production parking topology.

The curated records support a focused static-retrieval showcase,
not evidence of retrieval accuracy or complete application functionality. Demonstrating
hours, prices, and availability needs separate dynamic fixtures; approval and recording
need executable workflow scenarios, not more static records. Evaluate with a separate
question/reference-answer dataset, including paraphrases, multi-document questions,
and unknown answers. Increase the corpus when a planned demo question lacks coverage
or a retrieval test needs more realistic distractors, rather than to meet an arbitrary count.

Validate offline first, from the project root:

```bash
.venv/bin/python -m storage.seed --dry-run
```

Load the existing `.env` explicitly and embed/upsert the examples into the configured
Chroma collection (defaults to `parking_static` under `.data/chroma`):

```bash
uv run --env-file .env python -m storage.seed
```

This calls the configured embedding provider and requires the Azure/DIAL settings
described above. Application imports and the seed module do not load `.env` themselves.
The command validates the entire file before initializing storage. Repeating it updates
the same stable `demo-central:*` IDs rather than creating duplicates; unrelated IDs are
preserved, and SQLite is untouched. It does not delete previously seeded records omitted
from a newer file; delete obsolete IDs explicitly through `delete_documents`.

For retrieval, pass `location_id="demo-central"` to `static_store.search(...)`.
The chat graph still does not retrieve these documents automatically.

To expand the dataset, append reviewed records to the JSON array with unique,
stable IDs and the existing `id`, `page_content`, and `metadata` structure. There is no
fixed record-count limit. You can also supply a different reviewed file:

```bash
.venv/bin/python -m storage.seed --file path/to/reviewed-static.json --dry-run
uv run --env-file .env python -m storage.seed --file path/to/reviewed-static.json
```

The bundled file resolves relative to the seed module, independent of the working
directory; an explicit `--file` path resolves relative to your current working directory.
Keep each document within the embedding model's input limits; automatic chunking is
not implemented. Review custom text for sensitive information and source authenticity:
schema validation alone does not enforce either. Keep demo data out of production.

### Dynamic information: SQLite

`app.parking_agent.dynamic_store` exposes `upsert_fact(location_id, kind, value)`,
`get_fact(location_id, kind)`, `list_facts(location_id)`, and
`delete_fact(location_id, kind)`. Allowed kinds are `working_hours`, `prices`, and
`availability`. `parking_dynamic_facts` has a `(location_id, kind)` primary key,
a JSON value, and a UTC `updated_at` write timestamp. Missing facts return `None`
or an empty list, never defaults such as zero available spaces. Writes are atomic
upserts with parameter-bound SQL; connections are committed/rolled back and closed
per operation, rather than shared across graph workers.

JSON snapshots deliberately leave the domain schema open until real sources and
parking topology are decided. Payloads must be JSON-compatible, with string object
keys and finite numbers. `updated_at` is not a freshness guarantee; refresh rules
and upstream observation times remain future work. An availability snapshot is
**not** an allocation, hold, or reservation ledger. Both stores require explicit
location IDs without deciding whether the deployed system has one or many locations.

### Configuration and lifecycle

Optional environment settings are documented in `.env.example`:

| Setting | Default |
| --- | --- |
| `PARKING_CHROMA_DIRECTORY` | `.data/chroma` |
| `PARKING_CHROMA_COLLECTION` | `parking_static` |
| `PARKING_SQLITE_PATH` | `.data/parking.sqlite3` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | `DIAL_TEXT_EMBEDDING_3_SMALL` |
| `AZURE_OPENAI_EMBEDDING_MODEL` | `TEXT_EMBEDDING_3_SMALL` |

Relative paths are anchored to the project root, not the shell's current directory.
The CLI loads `.env`; direct Python callers must load/provide their environment
themselves. Existing `.env` values are unchanged. Run `uv sync --dev` to install the
updated lockfile. SQLite needs no extra package; Chroma needs no standalone server.
Chroma persists automatically and its local client follows the owning process's
lifetime. These synchronous storage APIs should run off the event loop when later
wired into async graph nodes. Back up the databases using database-supported backup
procedures; don't delete live files or copy a partially written directory.

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
The current prototype does not yet implement those services.

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
Seed checks also cover the 30-record fixture, extensibility, dry-run behavior, invalid
files/batches, repeatable ingestion, and preservation of unrelated documents.
They do not seed project data, make API calls, or evaluate live model behavior,
semantic retrieval accuracy, or production security enforcement.

Type, source lint, and formatting checks can run through uv's temporary tool environments
without changing the project dependencies:

```bash
uv tool run ty check
uv tool run ruff check app.py parking_agent.py commons storage tests
uv tool run ruff format --check app.py parking_agent.py commons storage tests
```

Ruff enables `BLE001` to prevent broad exception catches from returning.

References: [local server](https://docs.langchain.com/langsmith/local-server),
[Studio setup](https://docs.langchain.com/langsmith/quick-start-studio), and
[server-managed persistence](https://docs.langchain.com/oss/python/langgraph/persistence),
[Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma),
[persistent Chroma clients](https://docs.trychroma.com/docs/run-chroma/clients),
[Azure embeddings](https://reference.langchain.com/python/langchain-openai/embeddings/azure/AzureOpenAIEmbeddings),
and [SQLite](https://docs.python.org/3/library/sqlite3.html).
