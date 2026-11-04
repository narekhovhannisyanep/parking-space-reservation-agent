# Agent Guidelines

## Scope and collaboration

- Implement the original assignment's four stages, not a production-grade parking
  platform. [PROJECT_REQUIREMENTS.md](PROJECT_REQUIREMENTS.md) is the fixed baseline
  transcribed from the assignment screenshots; accepted additions and implementation
  choices below are separate from it. Do not turn examples, recommendations,
  optional improvements, extra credit, or existing code into mandatory requirements.
- Retain sensitive-data protection, secure and reliable recording resistant to
  unauthorized access, evaluation, and integration/load tests at assignment-appropriate
  scale. All four stages are implemented for the selected slice. Publication and
  submission are separate operator actions; do not expand scope without a decision.
- Preserve accepted behavior. If simplification would change it, or a consequential
  ambiguity, trade-off, or assumption arises, ask promptly. Explain the behavior and
  recommend an option so the user can decide. A discussion is not a decision. Exercise
  judgment on routine implementation details within agreed scope; do not silently make
  unagreed product decisions to avoid interruption.
- Before adding any third-party dependency, explain the problem, need, alternatives,
  and trade-offs, and obtain user approval. Approval of behavior is not package approval.
- Inspect actual Git status. Never stage, commit, push, publish, or submit autonomously.
  Do not modify the real `.env` or automatically reindex/delete populated project data.
  Preserve the explicit startup resets and isolated test paths described below.

## Assignment outcomes

Use Python, LangChain, LangGraph, and RAG with a vector database:

1. A parking chatbot answers general, hours, price, availability, and location questions;
   collects first name, surname, car number, and reservation period; filters sensitive
   data; and has a performance/accuracy evaluation report.
2. A second LangChain agent communicates the request to a human administrator and
   returns approval/refusal to the first agent.
3. An open-source file-writing MCP server or simple Python/FastAPI MCP server records
   only human-approved reservations as `Name | Car Number | Reservation Period | Approval Time`.
   The assignment permits a documented function-call fallback if MCP is not possible.
   Recording must be secure, reliable, and resistant to unauthorized access.
4. LangGraph integrates the stages; test dialogue, administrator confirmation, and
   recording end-to-end and under load; document architecture, logic, setup, usage,
   and deployment.

Provide a GitHub or EPAM GitLab repository link for each stage. Presentation, README,
at least two pytest/unittest tests per module, and CI/CD and/or infrastructure as code
are extra credit, not mandatory additions; see the baseline for details.

## Architecture and startup

- Two LangChain agents (parking and administrator communication) are coordinated by
  deterministic LangGraph/Python, without a third orchestrator LLM. Python owns
  allocation, state, approval enforcement, explicit submission, and recording. Only
  the human administrator approves; the recording service owns the file format.
- `ParkingSpaceReservationAgent` owns configuration, shared clients, storage helpers,
  and graph construction; graph threads own conversation/request state. `app:agent`
  and `app:webapp` share one agent; Agent Server provides checkpoints. Keep the
  response-node spelling `parking_assitant`. LangGraph CLI and LangSmith Studio are
  the entry point, not a terminal chat loop or additional public UI/service.
- Keep AzureChatOpenAI/DIAL and environment overrides unless the user chooses otherwise.
  Structured extraction defaults to `function_calling` and supports `json_schema`.
  The configured endpoint requires VPN for model/embedding requests.
- Python >=3.12. Imports and graph construction do not load `.env`, initialize clients,
  create databases, launch MCP, or make network calls. CLI loads `.env`; model clients
  remain lazy. Starlette supplies the server lifespan, not a separate public API.
  The exported graph rejects unprepared runs. Models, embeddings, stores, and
  checkpointers can be injected for offline tests.
- Startup prepares privacy detection, both databases, recording service/readiness, and
  bounded baseline evidence before interaction. Fail clearly on preparation errors;
  a printed CLI URL does not establish readiness. Validate recording paths before
  destructive resets; after resetting SQLite, recreate output, verify read-only ledger,
  private output and real stdio schemas, then prepare evidence. New threads and graph
  runs do not reset storage.
- Current flow: `reservation_collection`; information/options use
  `context_augmentation → parking_assitant`. Successful submission emits a pending
  reply, then `administrator_agent → administrator_review` interrupts in Studio.
  Approved review runs `record_approved_reservation` before `parking_assitant` reports
  both outcomes; refused/elapsed/unavailable review bypasses recording. Invalid human
  values reinterrupt. Separate checkpointed preparation/review nodes prevent resume
  from regenerating communication or creating another hold.

## Storage, inventory, and retrieval

- Persistent local Chroma holds public static descriptions and fixed mock USD/hour
  prices; built-in SQLite holds public hours/availability and private reservation state.
  No standalone database server is needed. Default paths are under project-root `.data/`;
  SQLite must not reside inside the persistent Chroma directory. Personal details and
  chats never enter either public-information store.
- Use the 20 individually identified spaces and their realistic differentiated route,
  benefit, and trade-off descriptions in `storage/parking_spaces.json`. Every space
  fits any vehicle by assumption; do not add vehicle categories, dimensions, or hard
  facility-feature matching. Descriptive preferences are semantic signals, not hard
  constraints or accessibility certification. Accessibility/store examples are not a
  mandatory feature checklist. `car_number` is the user's vehicle identifier, even for
  vehicles without plates; do not impose a plate format or invent one.
- Each fixture price is required integer `hourly_price_cents` in 100–2000, stored as
  numeric Chroma metadata, not embedded text. Do not randomize prices or duplicate their
  authority in SQLite. Legacy `prices` CRUD remains compatible but is excluded from
  evidence, filtering, and display. No locked quote, billing, total estimate, payment
  integration, or MCP/ledger price columns. Missing hours remain unknown until supplied
  through Python CRUD. Do not add facility addresses or booking-process content.
- Use schema 3 and collection `parking_spaces_v3`. Seed a missing/empty configured
  collection immediately at startup; reuse populated collections unchanged without
  freshness comparisons. Preserve older collections, refuse same-ID cross-schema
  upserts, and follow README migration instructions without erasing/reindexing data.
  Exact space filters retain location/type/schema scope.
- Reset SQLite empty once per application-server startup, including public snapshots and
  private ledger, not per thread. A failed dynamic read is not an empty database.
  Public availability accepts a location-wide boolean or space-ID/boolean map; `False`
  blocks, missing state/IDs leave static spaces available, and unsupported/null state
  fails selection. Snapshots do not allocate or guarantee period availability.
- Python deterministically retrieves evidence, not an LLM-selected tool. For availability
  and requests it derives a fixed public-topic query from the latest user message, filters
  eligible IDs and any budget before native top-3 similarity ranking and count, and never
  pads with over-budget results. Never put the entire inventory in the prompt. If a
  retriever cannot limit results, random selection may use only semantic candidates, not
  the whole inventory. IDs track spaces, not type/feature matching or arbitrary-ID search.
  Hours/prices-only turns omit inventory prose; price-only turns use a scoped metadata
  range. Generic turns use bounded startup evidence. Augment only a copy of the latest
  message; code appends every selected ID, description, and USD/hour rate to guarded
  replies. General LLM prose is prompt-grounded, not another price/option authority.
- Code-owned English topic vocabulary includes location, request and practical-benefit
  terms; negated stairs map to step-free terms, while "free of charge" is not a space
  request. Mask known option IDs before preference matching so `L1-`/`L2-` do not become
  level preferences. Relevance is determined by the existing structured extraction call,
  not a new classifier/agent: at most four prior privacy-guarded text messages of at most
  2,000 characters support topic follow-ups; field/budget proposals use only the latest
  sanitized text. Shared words alone do not establish parking relevance. Unrelated turns
  clear turn evidence and bypass retrieval, private updates and budgets while preserving
  draft, confirmation, options and budget. Greetings/scope clarification need no inventory
  evidence. Commands and submitted outcomes remain code-controlled. Model relevance and
  wording are not universal intent accuracy or an authorization boundary.

## Budget and conversation

- Budget is optional thread search state, not a fifth reservation field. The existing
  extraction call proposes a complete verbatim phrase; Python independently validates
  grounding, currency, units and exact cents and parses supported explicit English
  USD/hour upper limits and ranges. Under/below is strict; up-to/max and ranges are
  inclusive. Unsupported, ambiguous, conflicting or total-stay cues require clarification,
  not unrestricted suggestions; bounded grammar is not universal language understanding.
- Budget persists through `/options` and other-options; changes clear offers/round IDs,
  not personal confirmation. `/clear-budget`, `/new`, and `/cancel` clear it. Pending
  clarification blocks new offers and submission. Recheck the selected price before
  budgeted submission.
- Early browsing is allowed and suggestions never hold spaces. Preserve browse candidates
  and code-owned preferences through collection and duration changes. Confirmation
  automatically rechecks inventory, prices, public overrides and period overlap, retains
  eligible IDs/order, and explains/fills removals. `/options` optionally rechecks the same
  choices; other-options deliberately searches alternatives, excluding every ID offered in
  the current round, including temporarily ineligible IDs. Fresh requests, `/cancel`,
  and `/new` start a new round. Failed revalidation leaves no submittable stale binding.
- Collect first name, surname, user-provided vehicle identifier, and positive hours;
  never ask for dates, start/end times, or user timezone. `ValidatedReservationRequest`
  is collected details, not a submission or approval. Missing-detail submission prompts
  for missing fields; complete unconfirmed details get a confirmation summary. Legacy
  scheduled draft checkpoints retain personal fields but discard schedule/confirmation
  and require hours and fresh confirmation.
- Duration changes invalidate confirmation and offer-duration binding, not browse context;
  passing time alone does not invalidate confirmation/offers. Space references/fragments
  and negated values must not become vehicle identifiers; explicitly labeled arbitrary
  identifiers may match a space ID. Python checks grounded proposals and common English
  negative clauses. Clear or replace named incorrect fields. Vague disagreement clears
  confirmation, preserves draft fields, asks which detail to correct, and blocks `yes` and
  submission until clarified. Corrections are unavailable after submission. Bounded checks
  and model extraction do not guarantee universal semantic correctness.

## Submission and administrator review

- After confirmation and automatic option recheck, `/submit SPACE_ID` selects one currently
  offered ID. Python rechecks static inventory and any budgeted rate in Chroma before
  obtaining a SQLite write lock. It then rechecks public overrides and period overlap and
  inserts a pending hold atomically. Chroma and SQLite are not one transaction.
  Capture server UTC start inside the hold transaction and calculate end from confirmed
  hours. Waiting for review counts; approval never shifts the period. Offers preview a
  now-based window and bind to duration rather than an expiring draft start; submission
  checks its actual window. Unknown duration previews exclude active-now holds only.
- Half-open UTC periods `[start, end)` permit adjacent bookings. Pending and approved
  rows block overlap, including earlier requests in the same thread. Ended periods stop
  blocking on demand; no background expiry. Stable `draft_id:revision` identities make
  submission replay idempotent without reallocating or changing the original period.
  Old submitted checkpoints cannot resurrect rows cleared at startup.
- The second agent prepares bounded neutral review wording from canonical submitted
  details, not chat history; its wording omits interface names and request facts.
  Python supplies private facts and instructions through a typed Studio interrupt.
  The trusted local operator chooses only `approve` or `refuse`
  and clicks Resume: no default decision, manual request-ID entry, notification, or
  separate authenticated administrator service. Python resolves the checkpointed
  submission in the current thread and checks the internal review-payload ID against
  SQLite. Missing/invalid/extra response fields reinterrupt without changes; mismatches
  fail closed. Unsafe/unparseable model wording or API failure uses a fixed request.
- Refusal releases the hold. Approval before the fixed end retains the hold and UTC
  approval time; approval at/after end becomes `elapsed` and releases the hold. Decision
  replay is idempotent; reversal is out of scope. No editing, cancellation, or `/new`
  while awaiting review. After a decision, `/new` starts a separate one-space request;
  multiple spaces require sequential requests.

## Approved recording and protection

- SQLite is the authoritative operational ledger; approved-only files are output, not a
  second allocation source. Human approval and successful recording are separate outcomes.
  Failure retains approval and applicable allocation, reports failure/uncertainty honestly,
  and never claims a write without a validated success receipt. The initial attempt has
  at most one transient retry; `/retry-recording` retries only the current approved
  request before `/new`. A later ledger read failure must not claim no decision was
  applied. No ordinary-chat/background retry or cross-startup recovery.
- Use the official `mcp` SDK locked to 2.3.0 and a custom private stdio server. Python
  calls only `record_approved_reservation` with internal request/thread IDs; no LLM tool
  selection, arbitrary body/path/file, or approval flag. The server independently reads
  existing SQLite read-only, checks approved status and canonical details, and writes full
  name, vehicle identifier, original UTC start/end and original UTC approval instant.
  Previously valid approvals may be recorded after end; pending/refused/elapsed rows may
  not. Reject pipes/control-character injection while preserving Unicode and non-plate IDs.
- Each UTF-8 `.txt` file is named by SHA-256 of the internal request ID. Atomic
  no-overwrite publication and matching-content replay give idempotent file effects, not
  exactly-once delivery. Bounded replay checks are not allocation/ledger synchronization.
  Conflicts, unsafe paths/links and non-approved rows fail closed; temporary-name
  collisions cannot overwrite or delete another writer's data. Recording checkpoint
  results contain safe statuses/receipts, not raw personal fields or protocol exceptions.
- The dedicated disposable `approved-reservations/` directory is beside configured SQLite
  (`.data/approved-reservations/` by default); an SQLite override relocates it. Drop and
  recreate only this directory at server startup alongside SQLite reset, without a file
  manifest; never delete its parent or persistent Chroma. Use a fixed Python subprocess,
  no shell/network listener, provider-key inheritance or child `.env` loading. Directories
  are `0700`, SQLite/output/temporary files `0600`; cleanup is symlink-resistant and
  directory-descriptor scoped. Linux local POSIX primitives are required; unsupported
  primitives fail closed. Attempts have a 15-second deadline plus bounded SDK cleanup;
  timeout/cancellation closes subprocesses and diagnostic handles. Sanitize expected
  errors/SDK exception groups; propagate genuine programming errors and cancellation.

## Privacy, validation, and documentation

- Local `presidio-analyzer` and the pinned spaCy English model enforce privacy checks at
  public ingestion/evidence, model-copy and response boundaries, with rules/known values.
  Startup loads detection before databases; missing/broken detection fails closed and
  runtime never downloads models/domain data. Necessary reservation details can reach
  Azure/DIAL extraction and administrator composition; unrelated detected sensitive
  data is filtered, but extraction permits PERSON spans and may send unrelated names in
  the latest message. Own-draft summaries are code-generated. General-answer history and
  model output are sanitized; history excludes names/metadata on messages and arbitrary
  system/tool payloads. Thread `privacy_values` retains names/vehicle identifiers for
  filtering historical mentions after corrections/cancellation/new drafts. Guarded
  sync/async `messages` streams and administrator tokens use `nostream`.
- Studio, SQLite, checkpoints, raw chats, debug/state/task APIs, `astream_events`, and
  traces are trusted local development surfaces, not multi-user authorization or a
  sandbox. Unknown/obfuscated identifiers and unfamiliar PII may be missed; grounding
  and prompt-injection defense remain prompt-guided. Do not claim complete confidentiality
  or protection from the trusted account/root.
- Follow test-driven development for features/refactoring. Preserve offline startup,
  storage, retrieval, draft, review, recording and streaming tests; inject clients/stores
  and keep tests fast and offline by default and live-provider checks opt-in. Cover agreed
  requirements, not a fixed test count. `system_validation.py` uses isolated temporary
  Chroma, SQLite, output and checkpoints, verifies owned paths/readiness, omits offline
  provider keys and disables tracing.
  Stage 4 uses deterministic local full-workflow load and a small real Azure/DIAL pipeline
  smoke slice, with scripted decisions through the existing interrupt. Do not claim human
  throughput, live-provider load capacity, fresh browser validation, or production readiness.
  Optional retrieval/privacy tuning requires appropriate remeasurement.
- `python-pptx` 1.0.2 and Pillow 13.0.0 are isolated presentation tools pinned in
  `presentation/build_presentation.py`, not application dependencies or lockfile entries.
  Building the presentation must not load the application, real `.env`, or project databases. Current
  `presentation/` screenshots are approved presentation assets; do not label them bug
  evidence or require replacements. `presentation/demo.mp4` is separate, not embedded.
- This file applies repository-wide unless a more specific `AGENTS.md` applies. Keep
  durable project/collaboration instructions here, not in a separate memory instruction
  file. `CLAUDE.md` imports `@AGENTS.md`; do not duplicate context there. Keep
  `PROJECT_REQUIREMENTS.md` unchanged when documenting choices; record approved scope
  changes separately. Replace stale status rather than append development history.
- Documentation ownership: setup, commands and behavior in `README.md`; implemented
  fields in `PARKING_DATA_MODEL.md`; privacy evaluation in `GUARDRAIL_EVALUATION.md`;
  RAG evaluation in `RAG_EVALUATION.md`; full-system validation/load/live-smoke evidence
  and limits in `STAGE4_VALIDATION.md`; code style in `pyproject.toml`. Retain the MIT
  LICENSE and existing copyright notices.
