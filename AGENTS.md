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
- All four stages are implemented for the selected assignment slice. Stage 4's
  approved local-load plus small-live-smoke validation is documented below. Repository
  publication/submission is a separate operator action; do not expand scope or add
  dependencies without a new decision.

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
- **Static/dynamic split:** retain local persistent Chroma for static descriptions
  and the user-approved fixed mock hourly USD prices; built-in SQLite supplies hours,
  availability, and private reservation state. This supersedes the earlier SQLite-price
  choice. No standalone database server is needed.
- **Detailed mock inventory and budgets (user-approved addition):** keep the 20 space
  IDs/locations and realistic, differentiated route/benefit/trade-off descriptions.
  Accessibility/store examples are illustrative, not a mandatory feature checklist.
  Required integer `hourly_price_cents` is 100–2000 (USD/hour), stored with each
  description in the fixture but only as numeric Chroma metadata, not embedded text.
  No startup random pricing or duplicate authoritative SQLite prices. Legacy `prices`
  CRUD remains accepted for compatibility but is excluded from evidence/filtering/display.
  The existing structured extraction call proposes an optional complete verbatim
  budget phrase; Python independently parses supported explicit English USD/hour
  upper limits/ranges and validates grounding, units, currency and exact cent bounds.
  Under/below is strict, up-to/max inclusive, ranges inclusive. Unsupported/ambiguous/
  conflicting/total-stay price cues prompt clarification, not unrestricted suggestions.
  This bounded grammar is not universal semantic extraction accuracy. Budget is optional
  thread search state, not a fifth required reservation field. It persists through
  `/options`/other-options; changes clear offers/round IDs but not personal confirmation;
  `/clear-budget`, `/new`, `/cancel` clear it. Clarification pending blocks new offers/
  submission. Combine budget with available IDs in Chroma filters before native top-3
  ranking/result count, never pad with over-budget results. Code appends all selected
  IDs, descriptions and USD/hour rates to guarded replies. General LLM prose remains
  prompt-grounded, not an additional price/option authority. Recheck the selected rate
  before budgeted submission; no locked quote, billing, total estimate, payment integration
  or MCP/ledger price columns. Descriptive preferences are semantic signals, not hard
  feature constraints or accessibility certification. Use schema 3/new collection
  `parking_spaces_v3`; preserve older collections and refuse same-ID cross-schema upserts.
  Do not modify real `.env` or reindex/delete populated project data automatically.
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
  claim the file was written. The selected policy is one initial recording attempt plus
  at most one transient-failure retry, then `/retry-recording` for the current approved
  request before `/new`; no background retries or cross-startup recovery.
- **Recording output cleanup:** `.data/approved-reservations/` is dedicated entirely to
  disposable generated recording output. Drop/recreate that directory at application
  server startup alongside the SQLite reset; do not require a manifest of generated
  files. Keep deletion confined to this dedicated directory, never its parent or the
  static database. Per-reservation UTF-8 `.txt` files are selected, named by SHA-256 of
  the internal request ID. The dedicated directory is beside configured SQLite
  (`.data/approved-reservations/` by default); an explicit SQLite-path override relocates
  it too, so temporary test deployments never clear real project output.
- **Stage 3 MCP:** the user approved the official `mcp` SDK (locked to 2.3.0) and a
  custom private **stdio** server, not HTTP/FastAPI or the permitted function-call
  fallback. Python calls exactly one narrow `record_approved_reservation` tool with
  internal request/thread IDs. No LLM tool-selection adapter, arbitrary body/path/file
  argument or approval flag. The server independently reads existing SQLite in read-only
  mode, verifies approved status/canonical details and owns the four-field format: full
  name, vehicle identifier, original UTC start/end, original UTC approval instant.
  Atomic no-overwrite publication and matching-content replay provide idempotent file
  effects, not exactly-once delivery. Bounded output replay checks are not allocation
  or ledger synchronization. Conflicts, unsafe paths/links and non-approved rows fail
  closed; only a validated success receipt permits a recorded-success reply.
- **Recording protection:** fixed Python subprocess command, no shell/network listener,
  provider-key inheritance or child `.env` loading. Owner-only output directories
  (`0700`), output/temporary files and SQLite (`0600`); cleanup is symlink-resistant and
  directory-descriptor scoped. The tested deployment is Linux with local POSIX primitives;
  unsupported primitives fail closed. Attempts have a 15-second deadline plus bounded
  SDK cleanup. Timeout/cancellation closes subprocesses and diagnostic handles. Expected
  errors/SDK exception groups are sanitized; genuine programming errors/cancellation
  propagate. This is trusted-local protection, not a sandbox, separate human authentication,
  or confidentiality from the trusted account/root/Studio/debug APIs.
- **Conversation continuity/corrections (user-approved correction):** early browsing is
  allowed. Do not force `/options` after collection or silently replace useful choices.
  Preserve current browse candidates and code-owned descriptive preferences; confirmation
  automatically rechecks current inventory/prices/public overrides/period overlap,
  retains eligible IDs/order, and explains/fills necessary removals. `/options` optionally
  rechecks the same choices; "other options" deliberately searches alternatives. Keep
  all earlier IDs in the offer round, even when temporarily removed by eligibility.
  Duration changes invalidate confirmation/offer binding, not browse context. Space
  references/fragments and negated values must not become vehicle identifiers; explicitly
  labeled arbitrary vehicle identifiers may match a space ID. Named incorrect fields are
  cleared or replaced; vague disagreement clears confirmation and asks which detail needs
  correction, preserving draft fields and blocking `yes`/submission until clarified.
  Corrections remain unavailable after submission; bounded English checks and model
  extraction do not guarantee universal semantic correctness.
- **Stage 2 submission/holds:** confirm details, automatically recheck options, then select one
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
  business. The user-approved mock hourly prices/descriptive benefits above supersede
  the earlier free-request/directions-only choice. Do not add facility addresses,
  payment/cost accounting or booking-process content. Missing hours remain unknown
  until Python CRUD supplies them. The user picks one of up to three offered options,
  and the administrator approves or refuses it.
- **Sequential reservations:** multiple spaces in one conversation are separate
  sequential one-space requests, not one multi-space request, using `/new` after review.
- **Stage 4 evidence:** the user selected deterministic local full-workflow load plus
  a small real Azure/DIAL pipeline smoke slice, not measured live-provider load capacity.
  Scripted review responses use the existing typed interrupt; Studio remains the human
  channel. Do not claim human throughput, fresh browser validation, or production readiness.

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
  reply, then runs `administrator_agent → administrator_review` (typed Studio interrupt).
  Approved review runs `record_approved_reservation` before `parking_assitant`
  reports both outcomes. Refused/elapsed/unavailable review bypasses recording.
  Invalid human values loop
  to a fresh review interrupt. Model preparation and review are separate checkpointed
  nodes, so human resume does not regenerate communication or create another hold.
- Chroma static retrieval and SQLite JSON snapshot CRUD are implemented and graph-wired.
  Missing/empty Chroma collections are immediately seeded with 20 mock spaces at
  server startup; populated collections are reused unchanged. SQLite is reset empty
  once per server lifespan. Chat does not seed/reset either database. Default paths
  are project-root `.data/`; SQLite cannot reside inside the persistent Chroma directory.
- Static inventory currently uses schema version 3 and collection `parking_spaces_v3`.
  Exact space filters retain location/type/schema scope. See README for migration and
  PARKING_DATA_MODEL.md for implemented fields; do not erase existing static databases
  to migrate.
- Context retrieval uses fixed English public-topic queries instead of raw user text.
  The code-owned vocabulary includes location terms taken from fixture descriptions
  (stairs, outdoor area, levels, compass directions, wall, fence, lobby, walkway, corner)
  and request words (spot/place/park/vacant/options; "free of charge" is not a request).
  Benefit vocabulary covers step-free access/mobility, shopping bags, pushchairs,
  weather/sheltered routes, passing traffic and departures; negated stairs map to
  step-free terms. Exact budget constraints use Chroma numeric metadata, not embeddings.
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
  before ranking, so fewer than three options appear only when fewer eligible remain.
  Hours remain unknown when missing; prices come solely from Chroma metadata.
  Price-only turns read a scoped metadata range, not all inventory prose. These snapshots do not allocate
  or guarantee availability for a booking period. Stage 2 separately reads blocked IDs
  from the private ledger before ranking for a now-based preview of the validated
  duration; unknown durations use current observations and exclude only active-now holds.
  `offer_duration_hours` binds offered IDs to duration; duration changes invalidate that
  binding/confirmation but preserve browse candidates/public preferences until rechecked.
  Passing time alone does not invalidate confirmation/offers; submission rechecks overlap.
- Draft collection, field validation, and user confirmation are implemented.
  `ValidatedReservationRequest` means collected details, not a submission or approval.
  Current code uses positive duration hours, draft revisions, confirmation commands,
  and cancellation. `SubmittedReservationRequest` adds the server-generated UTC period
  only when holding. Missing-detail submission starts/continues collection with concrete
  missing-field prompts; complete unconfirmed details show a confirmation summary.
  Legacy draft checkpoints discard schedule/confirmation, preserve personal fields, and
  require duration plus fresh confirmation. See README for behavior.
  `ReservationExtraction` also proposes grounded `rejections` and `disagreement_phrase`;
  Python checks common English negative clauses independently, rejects space/negated
  field proposals, and owns field clearing and unresolved `correction_pending`. The
  graph retains `offer_candidate_ids`/code-only `offer_query` and revalidates on confirmation.
  Known option IDs are masked before preference matching so `L1-`/`L2-` do not become
  new level preferences. Failed automatic revalidation leaves no submittable stale IDs/binding.
- Private `storage/reservations.py` owns `parking_reservations` in the same SQLite
  database/lifecycle as public facts, but public CRUD/RAG never return its personal
  fields. Statuses are pending/approved/refused/elapsed. Stable `draft_id:revision`
  identity prevents duplicate/reallocated holds; write locking precedes eligibility
  reads. Startup clears both tables; old submitted checkpoints report missing requests
  rather than resurrect them. Stage 3 adds separate approved-only output/cleanup, not
  new recording fields/statuses in the operational ledger.
- `recording.py` owns output configuration, strict schemas, read-only canonical approval
  checks, formatting and atomic no-overwrite writes. It reuses submitted-request validation;
  pipes/control characters cannot inject fields/lines, Unicode/non-plate identifiers are
  preserved and valid prior approvals can be recorded after end. Replay leaves one complete
  file; conflicts and temporary-name collisions cannot overwrite/delete another writer's data.
- `recording_server.py` uses the official low-level SDK to expose exact extra-forbidden
  arguments and structured outcome/error codes. `recording_client.py` owns sync/native-async
  short-lived stdio sessions, startup schema validation, at most one transient retry and
  safe receipts. In-process MCP factories keep most offline graph fixtures fast; actual
  stdio is tested separately and through Agent Server. No raw MCP personal fields or
  exception/protocol contents enter recording checkpoint data or replies.
- The graph keeps `recording` as a JSON `RecordingResult` (recorded/failed/unconfirmed,
  identity, optional filename/error code, retryability, attempts) and `recording_requested`
  for explicit retry routing. `/new` clears the receipt. Failures preserve approval and
  allocation; transport/post-publication uncertainty is explicit. A later ledger read
  failure does not incorrectly claim no human decision was applied. No ordinary-chat
  auto-retry, background worker, approval reversal or period/timestamp shift is added.
- Startup validates recording paths before destructive resets, recreates the dedicated
  directory after SQLite reset, checks read-only ledger/private output and real stdio
  schemas, then prepares baseline evidence before serving. New threads/turns do not
  clear output. Imports do not launch MCP or create output directories.
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
- The detailed priced 20-record fixture preserves all original IDs/locations. Populated
  Chroma collections remain unchanged at startup; README documents a new schema-3
  collection migration and same-ID old-schema upserts are refused. Do not
  erase static data or silently apply fixture changes.
- GUARDRAIL_EVALUATION.md and guardrail_evaluation.py provide synthetic local privacy
  evaluation. Boundary tests cover contamination, detector failure, old drafts and
  actual guarded streaming.
- RAG_EVALUATION.md and rag_evaluation.py provide the Stage 1 evaluation report:
  original 20 labeled retrieval queries, 8 practical-benefit queries with 2 frozen
  out-of-vocabulary challenge paraphrases, 12 deterministic budget-compliance cases,
  request detection/follow-up disjointness, 13 rule-checked live answer conversations
  and latency. The 2026-11-02 real-embedding run showed mixed original-set changes:
  Recall@3 0.547, capped recall 0.625, Precision@3 0.500, Hit@3 0.750, MRR 0.567.
  Benefit capped recall/precision 0.583 and MRR 0.604; both challenge paraphrases missed.
  All 12 budget cases passed. Two live answer runs passed 12/13, with exact-ID lookup
  still failing; all offered options/prices now have code-generated entries.
  Five opt-in live component checks passed, including grounded upper-limit/range
  extraction. These are smoke/synthetic results, not universal language accuracy,
  hard feature matching, new browser validation or new live-provider load evidence.
- Offline tests cover storage, graph integration, draft collection, retrieval topics
  and evaluation harness. `system_validation.py` supplies explicit, bounded Stage 4
  scenarios/load/live checks through real localhost TCP Agent Server and actual stdio.
  Its child uses isolated temporary Chroma/SQLite/output/checkpoints, verifies owned
  storage paths and readiness, omits offline provider keys and disables tracing. No
  new public route, deployment service, dependency or administrator channel is added.
- `storage.__init__` lazily resolves/caches only the public `StaticParkingStore` export.
  The SQLite-only recorder no longer imports Chroma/provider clients. Public class
  identity/APIs, seeding/reuse, canonical recording checks and 15-second MCP deadlines
  are unchanged; dedicated import regressions protect that narrow optimization.
- Stages 1–3 are implemented; optional retrieval tuning remains possible. Stage 2 has
  offline sync/async interrupt/resume, concurrency, temporal, refusal, reset, replay and
  actual model-stream boundary tests. A network-blocked real Agent Server test verifies
  the decision-only schema served to Studio, empty-response retry and approval; the
  current extraction/administrator model schemas also have successful live smoke checks.
  The user reported successful VPN-connected manual Studio testing on 2026-10-31;
  individual browser scenarios were not itemized or independently checked by the coding agent.
  Stage 3 has 16 writer, 11 MCP and 9 recording-graph tests covering actual stdio,
  concurrent/replayed writes, invalid/unauthorized calls, failure before/after publication,
  grouped SDK errors, timeout/cancellation process/handle cleanup and manual retry.
  The network-blocked real Agent Server test now verifies decision-only resume followed
  by actual stdio MCP, exact file contents and final reply. The full offline run completed
  319 tests (4 opt-in live checks skipped) on 2026-10-31. These use synthetic data/injected
  models, not new browser/live-provider accuracy or full-system load evidence.
  Stage 4 now has separate evidence: 11 passing real HTTP/stdio end-to-end scenarios;
  90 passing local conversations at concurrency 1/5/10, with 45 approved files, no
  automatic recording retries/errors/overlaps or unreaped recorder children; a ten-way
  same-space race with one pending winner/nine expected conflicts; and two successful
  real Azure/DIAL approval/refusal pipeline smoke conversations. Live smoke used
  current function_calling, gpt-6-luna-2026-09-22 and text-embedding-3-small-1 with
  real temporary stores/MCP and no fallback review wording. Runs are dated 2026-10-31
  UTC (2026-11-01 local). See STAGE4_VALIDATION.md for methods/timings and limits:
  injected load models, scripted review, small samples/shared host, no live-provider
  load or new browser test, no complete privacy/security/resource-leak certification.
  The earlier Stage 4 offline regression completed 339 tests (335 passed, four live
  checks skipped); the two full-pipeline live checks remain separate evidence.
  The 2026-11-02 priced-inventory regression completed 359 tests (354 passed, five
  opt-in live component checks skipped), including a budget-filtered real HTTP/stdio
  approval/recording check. Ruff lint/format and ty checks passed. The fixture/pricing
  changes are the user-approved addition above; no dependency/provider or baseline
  requirement change, real `.env` edit, or populated project-data reset was made.
  The conversation-fix regression completed 389 tests (382 passed, seven opt-in live
  checks skipped) in 329.240 s. All seven live components, 13 HTTP/stdio scenarios,
  two repeated real-provider screenshot approval/refusal flows, six local conversations
  at concurrency five, and refreshed ten-way contention passed. See STAGE4_VALIDATION.md
  for separate historical benchmark/current regression evidence and limitations.

## Next-session handoff

- **Where we stopped:** All four stages use duration-only requests → confirmation/automatic option recheck
  → `/submit SPACE_ID` → atomic SQLite hold → second LangChain communication agent →
  decision-only typed Studio review → approved-only stdio MCP → code-controlled outcome.
  No human request-ID copying or default decision. Period starts at submission; waiting
  counts. Valid prior approval is retained if recording fails; `/retry-recording` repeats
  only current-request recording before `/new`. Output directory is deliberately reset
  alongside SQLite on next startup; old checkpoints cannot resurrect rows/files.
  Recording choices remain settled. README documents setup, logic, Studio use,
  trusted-local deployment and repository links; STAGE4_VALIDATION.md documents the
  completed integration/local-load/live-smoke slice. Detailed mock descriptions,
  fixed USD/hour metadata and optional persistent budgets are now implemented.
  Screenshot-discovered regressions in option continuity, space-versus-vehicle extraction,
  and negative/vague corrections have been fixed; `/options` is now optional. The
  requested dark English PowerPoint (~12–14 slides/10 minutes, author Narek Hovhannisyan)
  awaits replacement demo screenshots in `presentation/` after the user's fresh Studio test.
  Preserve the existing bug screenshots as diagnostic evidence, not successful demo evidence.
  For the earlier demo, the operator must select `parking_spaces_v3` in the real
  `.env` and seed the new collection explicitly or at startup as documented in README;
  real `.env` and populated project data have not been changed automatically.
  No provider/dependency change was needed; do not erase/reseed populated Chroma.
  Small live pipeline checks do not
  broaden the earlier browser/demo evidence or establish real-provider load capacity.
- **First action:** inspect actual Git status and the user's next request. If asked,
  review/commit/publish the completed assignment and use README's repository links
  for the stage answer fields. Do not stage, commit, push or submit automatically.
  Optional presentation/CI/retrieval tuning is not missing core implementation;
  no production platform or replacement administrator channel has been selected.
- Preserve the existing offline startup, storage, retrieval, and draft tests. Obtain
  approval before adding any dependency, and keep setup/status documentation aligned
  with each implemented slice. Inspect actual git status rather than assuming changes
  from a prior session are still uncommitted; do not stage or commit without instruction.

## Remaining decisions and work

- Optional: tune retrieval weak spots listed in RAG_EVALUATION.md and re-measure.
  Broaden privacy evaluation as needed; the synthetic detector report is not complete confidentiality,
  authorization, or live-model correctness.
- Preserve the selected Stage 3 MCP/layout/retry/protection behaviors. A user acknowledgment
  is not human approval; pending/refused/elapsed requests cannot be recorded, and approval
  alone does not prove a successful write. Keep SQL approval/allocation intact on failures.
- Retain startup readiness, static reuse/seeding, SQLite reset, three-option suggestions,
  and their offline tests when modifying the workflow.
- Preserve Stage 2's accepted explicit-submit, period, refusal, waiting, reset, and
  replay rules and its offline/live smoke checks. Additional itemized browser scenarios
  are optional validation, not an unimplemented Stage 2 feature.
- Core integration/load validation and assignment documentation are complete for the
  selected slice. Repository publication/answer-field submission remains an operator
  action. Do not create a backlog of hypothetical production UI, tenancy, scaling,
  observability or infrastructure work.

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
  limits: RAG_EVALUATION.md. Full-system validation/load/live-smoke evidence and limits:
  STAGE4_VALIDATION.md.
  Code style: pyproject.toml. Retain the MIT LICENSE and existing copyright notices.
