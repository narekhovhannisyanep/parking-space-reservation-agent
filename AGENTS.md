# Agent Guidelines

## User intent and scope — read first

- Implement the original assignment's four stages, **not a production-grade parking
  platform**. The user wants the stated requirements implemented without speculative
  scaling, future-proofing, or extra features. Scope grows only when the user requests it.
- The original requirements are transcribed in [PROJECT_REQUIREMENTS.md](PROJECT_REQUIREMENTS.md)
  from the five assignment screenshots supplied by the user. Treat that baseline as
  fixed; keep user-approved additions and implementation choices separate from it.
- Keep the assignment's sensitive-data protection, secure/reliable recording service,
  resistance to unauthorized access, evaluation, and load/integration tests. Meet them
  with a practical solution appropriate to this assignment, not enterprise infrastructure.
- Do not turn examples, recommendations, optional improvements, or extra-credit
  artifacts into mandatory requirements. Do not turn existing code into a specification.
- Preserve accepted choices below. If simplification would change one, or a choice is
  genuinely unclear, ask the user before proceeding. A discussed idea is not a decision.
- **Questions and discussion are welcome:** the user treats design discussions as a
  learning opportunity. Ask promptly whenever ambiguity, a trade-off, or a consequential
  assumption arises; do not silently guess to avoid interrupting progress. Explain the
  relevant behavior and your recommendation so the user can make an informed decision.
  Exercise judgment on routine implementation details within agreed scope, but do not
  interpret trust in that judgment as permission to make unagreed product decisions.
- **New dependencies:** before adding a third-party package, explain what problem it
  solves, why it is needed, and relevant alternatives or trade-offs, then obtain the
  user's approval. Approval of a behavior does not automatically approve a new package.
- Stage 2 submission/holds/administrator review are now selected and implemented below.
  Stage 3 recording remains separate; discuss its remaining choices before implementing it.

## Required outcomes

Use Python, LangChain, LangGraph, and RAG with a vector database.

1. **RAG chatbot:** answer general parking, hours, prices, availability, and location
   questions; interactively collect first name, surname, car number, and reservation
   period; implement sensitive-data guardrails/filtering; evaluate performance and
   accuracy and provide an evaluation report.
2. **Human-in-the-loop:** a second LangChain agent contacts the human administrator
   after details are collected, sends the reservation request, receives approval or
   refusal, and maintains communication with the first agent.
3. **Approved recording:** integrate an open-source file-writing MCP server or a
   simple Python/FastAPI MCP server. If MCP is not possible, the assignment permits
   a documented tool/function-call fallback. Only human-approved reservations are
   written to a text file as `Name | Car Number | Reservation Period | Approval Time`.
   The recording service must be secure, reliable, and resistant to unauthorized access.
4. **LangGraph integration:** connect all stages into a stable workflow; test it
   end-to-end and load-test chatbot dialogue, administrator confirmation, and recording;
   document architecture, agent/server logic, usage, setup, and deployment.

Provide a GitHub or EPAM GitLab repository link for each stage. Presentation,
README, at least two pytest/unittest tests per module, and CI/CD and/or infrastructure
as code are extra-credit artifacts; see the baseline for details.

## Accepted choices to retain

- **Two agents, no third orchestrator LLM:** the user-facing parking agent and the
  administrator-communication agent are coordinated by deterministic LangGraph/Python
  code. This keeps the approval workflow explicit without another model call.
- **Python responsibility:** allocation, reservation state, approval enforcement, and
  record formatting/writing belong in code, not LLM decisions. Agents collect/clarify
  details and coordinate communication; only the human administrator approves.
  The recording service owns the file format, not the user-facing system prompt.
  Explicit `/submit SPACE_ID` is code-controlled; agents do not decide allocation.
- **Static/dynamic split:** retain the selected optional assignment improvement:
  local persistent Chroma for static information and built-in SQLite for dynamic
  hours, prices, and availability. No standalone database server is needed.
- **Minimal space inventory:** individually identified spaces; every space fits any
  vehicle by project assumption. No vehicle categories, dimensions, or facility-feature
  modeling. `car_number` is a user-provided vehicle identifier, including for vehicles
  without plates; do not require a plate format or invent an identifier.
- **Static seed and reuse:** the bundled `storage/parking_spaces.json` mock inventory
  is approved for initial seeding. At server startup, open the configured Chroma
  collection and immediately seed it if missing or empty. Reuse a populated collection
  unchanged; do not compare seed contents, timestamps, or freshness on each run.
  A static-data-change/approval workflow is no longer in scope.
- **Startup readiness:** before accepting user interaction, both databases and bounded
  baseline evidence must be ready. Clear/reinitialize SQLite empty at each application
  server startup, not at each thread or graph run. Startup preparation failures must
  stop startup with a clear exception, not leave a superficially working chatbot.
- **Availability suggestions:** static inventory spaces are available by default unless
  applicable dynamic state blocks them. Read dynamic data when needed; a failed read
  is not an empty database. For availability/request topics, Python derives a fixed
  public-topic query from the latest user message, applies dynamic eligibility, and
  retrieves up to three similarity-ranked eligible records using Chroma embeddings.
  Native result limits are supported, so no random sampling is needed. If a retriever
  cannot limit results, random selection may only use its semantic candidates, never
  the whole inventory. An LLM does not decide eligibility. Do not include all inventory
  in the prompt. Keep offered IDs in
  thread state without writing/holding spaces merely because they were suggested.
  Space IDs are needed for tracking, not size/type/feature matching or arbitrary-ID search.
- **Reservation state and recording:** SQLite is the intended authoritative operational
  reservation state; the Stage 3 text file is an approved-only output, not a second
  allocation source to synchronize or read back during normal operation. Approval
  reversal is not in scope. Human approval and successful file writing remain separate
  outcomes. If file writing fails, retain the human-approved reservation and its
  applicable allocation in SQLite, explicitly report recording failure, and never
  claim the file was written. Retry behavior remains to be agreed during Stage 3.
- **Recording output cleanup:** `.data/approved-reservations/` is dedicated entirely to
  disposable generated recording output. Drop/recreate that directory at application
  server startup alongside the SQLite reset; do not require a manifest of generated
  files. Keep deletion confined to this dedicated directory, never its parent or the
  static database. Single-file versus per-reservation output remains undecided. This
  choice may be revisited if Stage 3 reveals a concrete need, not hypothetical future
  requirements; do not implement recording or its cleanup in the startup/suggestions
  change.
- **Stage 2 submission/holds:** confirm details, request `/options`, then select one
  currently offered ID with `/submit SPACE_ID`. Python atomically rechecks public
  overrides and period overlap and inserts a pending SQLite hold before review.
  Offers never hold spaces. Half-open UTC periods `[start, end)` allow adjacent bookings;
  pending/approved rows block overlaps, including earlier requests from the same thread.
- **Immediate duration-only reservations (user-approved replacement for scheduled booking):**
  collect first name, surname, vehicle identifier, and how many hours; never ask for
  dates, start/end times, or the user's timezone. Python uses the server UTC clock.
  The user selected submission time, not approval time, as the start: capture it inside
  the atomic hold transaction and calculate end from the confirmed duration. Waiting
  for administrator review counts toward those hours. Offers preview a now-based window
  and bind to duration, not an expiring draft start; submission rechecks its actual window.
- **Administrator channel:** the user chose Studio review, not REST/email/messenger.
  The second LangChain agent prepares a bounded review message from canonical submitted
  details, and Python shows the private facts through a typed LangGraph interrupt.
  The trusted local human operator selects only `approve`/`refuse` and clicks Resume;
  no manual request-ID entry or default decision. Python resolves the checkpointed
  submission in the current thread and checks its internal review-payload ID against
  the SQLite record. Internal IDs remain for tracking/idempotency, not human input.
  Studio is not a separate authenticated administrator application; no notification,
  public UI, new service, or new dependency is selected.
- **Decision lifecycle:** refusal immediately releases the hold; approval retains it
  and a UTC approval time. No user editing/cancellation or `/new` while awaiting review.
  Approval before the fixed end is allowed; approval at/after end becomes `elapsed`, not
  approved, and releases the hold. Ended periods stop blocking on demand; no automatic
  expiry timer/background cleanup. `/new` after a decision starts a separate request.
  Decision/submission replay is idempotent; approval reversal is out of scope.
- **Entry point:** LangGraph CLI + LangSmith Studio, not a terminal chat loop.
  No additional public UI or production deployment platform has been requested.
- **Agent ownership:** `ParkingSpaceReservationAgent` owns configuration, shared clients,
  storage helpers, and graph construction. Conversation/request state belongs to graph
  threads. `app.py` exports `agent`; Agent Server supplies checkpoint persistence.
  Keep the user-selected response-node spelling `parking_assitant`.
- **Current provider:** retain the existing AzureChatOpenAI/DIAL configuration and
  environment overrides unless the user decides to change it. Provider replacement
  is not a prerequisite for finishing the assignment.
- **Deterministic retrieval:** Python retrieves evidence before information answers;
  retrieval is not an LLM-selected tool. Personal reservation details and chats are
  not ingested into either public-information store.
- **Approved privacy slice:** local presidio-analyzer and a pinned spaCy English model
  are approved dependencies. Necessary reservation details may still reach Azure/DIAL
  extraction; unrelated detected sensitive data is filtered. General-answer history
  and model outputs are sanitized; own-draft summaries remain code-generated. Studio,
  checkpoints and debug/state APIs remain trusted local development surfaces, not a
  multi-user authorization boundary. Do not claim complete confidentiality.
- **Learning-project focus:** this is a RAG/LangGraph learning project, not a parking
  business. Do not add facility address, cost/pricing models, or booking-process
  content. Space descriptions are the public location evidence; hours/prices remain
  unknown unless supplied through Python CRUD. Requests are free: the user picks one
  of the three offered options, and the administrator approves or refuses it.
- **Sequential reservations:** multiple spaces in one conversation are separate
  sequential one-space requests, not one multi-space request, using `/new` after review.

## Current implementation, not additional requirements

- Python >=3.12. Imports/graph construction do not load `.env`, initialize clients,
  create databases, or make network calls. CLI loads `.env`; the server lifespan prepares
  storage/evidence before requests, while model clients remain lazy. Module exports
  `app:agent` and `app:webapp` share one agent. The exported graph rejects unprepared runs.
  Models, embeddings, stores, and checkpointers can be injected for offline tests.
- Starlette supplies the custom server lifespan, not a separate public API/UI. With the
  current endpoint, VPN connectivity is needed for embedding/model requests. Mandatory
  first-time seeding waits on embeddings before serving Studio; a printed CLI URL is
  not proof of readiness. Studio connectivity was restored by connecting the VPN,
  without a code or provider change; this does not establish live-model accuracy.
- Current graph starts with `reservation_collection`; information/options use
  `context_augmentation → parking_assitant`. Successful submission emits the pending
  reply, then runs `administrator_agent → administrator_review` (typed Studio interrupt)
  and returns to `parking_assitant` for the actual outcome. Invalid human values loop
  to a fresh review interrupt. Model preparation and review are separate checkpointed
  nodes, so human resume does not regenerate communication or create another hold.
- Chroma static retrieval and SQLite JSON snapshot CRUD are implemented and graph-wired.
  Missing/empty Chroma collections are immediately seeded with 20 mock spaces at
  server startup; populated collections are reused unchanged. SQLite is reset empty
  once per server lifespan. Chat does not seed/reset either database. Default paths
  are project-root `.data/`; SQLite cannot reside inside the persistent Chroma directory.
- Static inventory currently uses schema version 2 and collection `parking_spaces_v2`.
  Exact space filters retain location/type/schema scope. See README for migration and
  PARKING_DATA_MODEL.md for implemented fields; do not erase existing static databases
  to migrate.
- Context retrieval uses fixed English public-topic queries instead of raw user text.
  The code-owned vocabulary includes location terms taken from fixture descriptions
  (stairs, outdoor area, levels, compass directions, wall, fence, lobby, walkway, corner)
  and request words (spot/place/park/vacant/options; "free of charge" is not a request).
  Retrieval reads location-scoped dynamic facts, and augments only a copy of the latest user
  message. Hours/prices-only turns omit space inventory; generic turns reuse bounded
  startup static evidence. Availability/request turns semantically retrieve up to three
  scoped spaces using the message-derived topic query, constrained to IDs eligible
  under successfully read dynamic overrides. They do not sample the entire inventory.
  Existing
  public availability snapshots support a boolean or space-ID/boolean map (`False`
  blocks); absent state leaves static spaces available by default. Unsupported/null
  snapshots fail selection. Offered IDs stay thread-scoped; suggestions do not hold
  spaces. "Other/different/more options" follow-ups exclude every ID offered in the
  current round (`offer_round_ids`, code-tracked, never model text) before ranking; a
  fresh request, `/cancel`, or `/new` starts a new round. Blocked spaces are filtered
  before ranking, so fewer than three options appear only when fewer eligible remain. Hours/prices remain unknown when missing. These snapshots do not allocate
  or guarantee availability for a booking period. Stage 2 separately reads blocked IDs
  from the private ledger before ranking for a now-based preview of the validated
  duration; unknown durations use current observations and exclude only active-now holds.
  `offer_duration_hours` binds offered IDs to duration; duration changes invalidate offers.
  Passing time alone does not invalidate confirmation/offers; submission rechecks overlap.
- Draft collection, field validation, and user confirmation are implemented.
  `ValidatedReservationRequest` means collected details, not a submission or approval.
  Current code uses positive duration hours, draft revisions, confirmation commands,
  and cancellation. `SubmittedReservationRequest` adds the server-generated UTC period
  only when holding. Missing-detail submission starts/continues collection with concrete
  missing-field prompts; complete unconfirmed details show a confirmation summary.
  Legacy draft checkpoints discard schedule/confirmation, preserve personal fields, and
  require duration plus fresh confirmation. See README for behavior.
- Private `storage/reservations.py` owns `parking_reservations` in the same SQLite
  database/lifecycle as public facts, but public CRUD/RAG never return its personal
  fields. Statuses are pending/approved/refused/elapsed. Stable `draft_id:revision`
  identity prevents duplicate/reallocated holds; write locking precedes eligibility
  reads. Startup clears both tables; old submitted checkpoints report missing requests
  rather than resurrect them. No Stage 3 file output/cleanup is implemented.
- `administrator_agent.py` uses its own lazy LangChain structured-output runnable
  and prompt on the existing Azure/DIAL client. It receives necessary submitted details,
  not chat history. Generated wording omits interface names and request facts; Python
  supplies review fields and UI instructions separately. This avoids the local detector's
  PERSON false positive on “Studio” without weakening guards. Python owns payload facts
  and decisions; unsafe/unparseable wording
  or API failure uses a fixed review request. Raw administrator tokens use `nostream`.
  Private payloads, SQLite, checkpoints/debug and Studio remain trusted local data.
  The typed human response has only required `decision`; missing/invalid/extra fields
  reinterrupt without changing SQLite. An internal review/request mismatch fails closed
  before showing/applying a decision. These checks are not administrator authentication.
- Structured extraction defaults to `function_calling`, with explicit `json_schema`
  support. Four opt-in synthetic live smoke checks passed on 2026-10-31 with
  `function_calling` and `gpt-6-luna-2026-09-22` for current duration-only extraction/
  multi-turn collection and sync/async administrator structured output. The administrator
  checks inspect parsed output before fallback and verify accepted wording plus unchanged
  canonical facts. Mocked stores and disabled tracing isolate these checks; they are not
  live full-workflow, accuracy, load, or security tests.
- Privacy checks are code-enforced at public ingestion/evidence, model-copy and response
  boundaries using local Presidio/spaCy plus rules/known values. Imports do not load NLP;
  startup prepares it before databases. Missing/broken detection fails closed; runtime
  never downloads models/domain data. General model history excludes message names,
  arbitrary metadata/system/tool payloads; necessary private extraction remains.
  `privacy_values` retains thread-local names/vehicle identifiers for filtering history
  after corrections/cancellation/new drafts. Guarded sync/async `messages` streaming
  suppresses raw model tokens with `nostream`. Debug/state/task APIs, `astream_events`,
  traces, raw chats, and checkpoints remain trusted private development data. Unknown/
  obfuscated identifiers and unfamiliar PII can be missed. Information grounding and
  prompt-injection defense remain prompt-guided; do not claim complete confidentiality.
- The 20-record fixture preserves the original eight IDs. Populated Chroma collections
  remain unchanged at startup; README documents a one-time explicit seed upsert. Do not
  erase static data or silently apply fixture changes.
- GUARDRAIL_EVALUATION.md and guardrail_evaluation.py provide synthetic local privacy
  evaluation. Boundary tests cover contamination, detector failure, old drafts and
  actual guarded streaming.
- RAG_EVALUATION.md and rag_evaluation.py provide the Stage 1 evaluation report:
  20 labeled retrieval queries (Recall@3/capped recall/Precision@3/Hit@3/MRR, measured
  before and after the vocabulary change with real embeddings), request detection,
  follow-up disjointness, 13 rule-checked live answer conversations, and per-node
  latency. Answers use rules, not an LLM grader. Remaining weak spots (lift/elevator
  synonyms, one-word topics, exact-ID lookup, not all offered options mentioned by
  the model) are documented, not tuned, to avoid overfitting the small labeled set.
- Offline tests cover storage, graph integration, draft collection, retrieval topics
  and evaluation harness. Required load tests are not complete (Stage 4).
- Stages 1–2 are implemented; optional retrieval tuning remains possible. Stage 2 has
  offline sync/async interrupt/resume, concurrency, temporal, refusal, reset, replay and
  actual model-stream boundary tests. A network-blocked real Agent Server test verifies
  the decision-only schema served to Studio, empty-response retry and approval; the
  current extraction/administrator model schemas also have successful live smoke checks.
  The user reported successful VPN-connected manual Studio testing on 2026-10-31;
  individual browser scenarios were not itemized or independently checked by the coding agent.
  Stages 3–4 remain incomplete: MCP/file recording and full-pipeline/load tests.

## Next-session handoff

- **Where we stopped:** Stages 1–2 use duration-only immediate reservations: names,
  vehicle identifier, hours → confirmed details →
  `/options` → `/submit SPACE_ID` → atomic private SQLite hold → second LangChain
  communication agent → decision-only typed Studio human review → code-controlled outcome reply.
  The administrator reviews the fields and chooses approve/refuse → Resume; no request
  ID is copied or constructed. Internal request identity is checked by Python. Studio
  can flag an empty Decision immediately; no default decision is supplied to hide that.
  Python fixes the UTC start/end at submission; review waiting consumes the duration,
  and approval is allowed until end. Missing-field prompts and offline delay/replay/
  boundary tests are updated. README contains the demo and trusted-local limitations.
  Four synthetic live smoke checks passed on 2026-10-31 for the current extraction
  and sync/async administrator model schemas, including accepted wording before fallback.
  The user reported successful manual Studio testing with the VPN connected; browser
  scenarios were not itemized or independently checked by the coding agent. Offline
  tests cover the approval/refusal graph and real Agent Server schema; live smoke tests
  do not exercise a complete reservation/hold/human-resume pipeline. Stage 2 is ready
  to wrap up for the selected scope. No static data or provider change was needed;
  do not erase/reseed populated Chroma.
- **First action:** agree on the Stage 3 recording slice with the user: MCP implementation
  (or documented permitted fallback), output-file layout, retries, and practical access
  protection. Do not substitute REST/email/messenger for the selected Studio administrator
  channel without user agreement.
- **Before the recording slice:** settle MCP implementation (or documented permitted
  fallback), file layout, retries, and access protection. Then implement approved-only
  recording and dedicated output-directory startup cleanup, preserving SQLite approval
  and allocation if writing fails. These behaviors are agreed; their detailed wiring
  is not. Finish full integration and end-to-end/load testing after connecting the flow.
- Preserve the existing offline startup, storage, retrieval, and draft tests. Obtain
  approval before adding any dependency, and keep setup/status documentation aligned
  with each implemented slice. Inspect actual git status rather than assuming changes
  from a prior session are still uncommitted; do not stage or commit without instruction.

## Remaining decisions and work

- Optional: tune retrieval weak spots listed in RAG_EVALUATION.md and re-measure.
  Broaden privacy evaluation as needed; the synthetic detector report is not complete confidentiality,
  authorization, or live-model correctness.
- Choose MCP implementation (or permitted fallback), output-file layout and retry
  behavior, and the practical access protection needed by the recording service.
- Connect the implemented code-controlled administrator approval to recording.
  A user acknowledgment is not administrator approval; pending
  or refused requests must not be recorded, and approval alone does not prove a write.
- Retain startup readiness, static reuse/seeding, SQLite reset, three-option suggestions,
  and their offline tests when implementing subsequent workflow stages.
- Preserve Stage 2's accepted explicit-submit, period, refusal, waiting, reset, and
  replay rules and its offline/live smoke checks. Additional itemized browser scenarios
  are optional validation, not an unimplemented Stage 2 feature.
- Finish integration/load tests and assignment documentation. Do not create a backlog
  of hypothetical production UI, tenancy, scaling, observability, or infrastructure work.

## Development practice

- Follow test-driven development for new features and refactoring: write or adjust
  a failing test that states the intended behavior, then implement until it passes.
- Cover the main project requirements and agreed behaviors, not a fixed test count
  per module. Keep tests fast and offline by default (inject models, embeddings,
  stores); live-provider checks stay opt-in. Avoid tests that slow development
  without protecting a requirement or an agreed choice.

## Maintaining documentation

- This file applies repository-wide unless a more specific AGENTS.md applies.
- Keep scope, accepted choices, current status, and real undecided work here. Replace
  stale information; do not append implementation history or speculative roadmaps.
- Keep PROJECT_REQUIREMENTS.md unchanged when documenting implementation choices.
  Record a user-approved scope change separately and explicitly.
- CLAUDE.md imports @AGENTS.md; do not duplicate project context there.
- The user intentionally removed `.github/instructions/memory.instruction.md`.
  Keep persistent project/collaboration instructions in `AGENTS.md`; do not recreate
  a separate memory instruction file.
- Setup, commands, and behavior: README.md. Implemented data fields: PARKING_DATA_MODEL.md.
  Synthetic privacy results/limits: GUARDRAIL_EVALUATION.md. RAG evaluation results/
  limits: RAG_EVALUATION.md.
  Code style: pyproject.toml. Retain the MIT LICENSE and existing copyright notices.
