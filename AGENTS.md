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
- The database-startup/availability-suggestions implementation is deliberately limited
  to preparation and options. Reservation holds, administrator decisions, and recording
  are subsequent work; discuss their remaining choices before implementing them.

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
  Submission/allocation may be exposed through tools; detailed wiring is not decided.
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
- **Later request holds:** a requested space must be recorded in SQLite as held before
  administrator review, and human approval changes its reservation status. Retain
  reservation period and approval time. Detailed selection/hold timing, overlap rules,
  and on-demand validation/cleanup remain to be agreed when implementing this flow;
  do not invent hold expiry timers or background cleanup infrastructure.
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

## Current implementation, not additional requirements

- Python >=3.12. Imports/graph construction do not load `.env`, initialize clients,
  create databases, or make network calls. CLI loads `.env`; the server lifespan prepares
  storage/evidence before requests, while model clients remain lazy. Module exports
  `app:agent` and `app:webapp` share one agent. The exported graph rejects unprepared runs.
  Models, embeddings, stores, and checkpointers can be injected for offline tests.
- Current graph: `START → reservation_collection`; information turns continue through
  `context_augmentation → parking_assitant → END`. Deterministic collection replies
  go directly to `parking_assitant → END` without retrieval, except incomplete request
  turns mentioning availability/reservation, which retrieve options before replying.
- Chroma static retrieval and SQLite JSON snapshot CRUD are implemented and graph-wired.
  Missing/empty Chroma collections are immediately seeded with eight mock spaces at
  server startup; populated collections are reused unchanged. SQLite is reset empty
  once per server lifespan. Chat does not seed/reset either database. Default paths
  are project-root `.data/`; SQLite cannot reside inside the persistent Chroma directory.
- Static inventory currently uses schema version 2 and collection `parking_spaces_v2`.
  Exact space filters retain location/type/schema scope. See README for migration and
  PARKING_DATA_MODEL.md for implemented fields; do not erase existing static databases
  to migrate.
- Context retrieval uses fixed English public-topic queries instead of raw user text,
  reads location-scoped dynamic facts, and augments only a copy of the latest user
  message. Hours/prices-only turns omit space inventory; generic turns reuse bounded
  startup static evidence. Availability/request turns semantically retrieve up to three
  scoped spaces using the message-derived topic query, constrained to IDs eligible
  under successfully read dynamic overrides. They do not sample the entire inventory.
  Existing
  public availability snapshots support a boolean or space-ID/boolean map (`False`
  blocks); absent state leaves static spaces available by default. Unsupported/null
  snapshots fail selection. Offered IDs stay thread-scoped; suggestions do not hold
  spaces. Hours/prices remain unknown when missing. These snapshots do not allocate
  or confirm availability for a booking period; a future hold ledger is not implemented.
- Draft collection, field validation, and user confirmation are implemented.
  `ValidatedReservationRequest` means collected details, not a submission or approval.
  Current code uses explicit dates/timezones, UTC normalization, draft revisions,
  confirmation commands, and cancellation. See README for behavior; these details
  may be simplified through a separately agreed refactor.
- Structured extraction defaults to `function_calling`, with explicit `json_schema`
  support. Synthetic live checks timed out; endpoint compatibility remains unverified.
- Information grounding/general privacy guardrails are still prompt-guided, not fully
  enforced. Checkpoints/Studio can expose private draft state; do not claim data
  protection is complete merely because public graph schemas expose only messages.
- Offline tests cover storage, graph integration, and draft collection. Live model
  accuracy, RAG evaluation/reporting, and required load tests are not complete.
- All four stages remain incomplete. Submission, administrator communication/decision
  handling, allocation, MCP/file recording, and the complete pipeline are not implemented.

## Remaining decisions and work

- Complete Stage 1 information coverage, required guardrails, and evaluation.
- Choose the administrator communication channel and simple approval/refusal workflow.
- Choose MCP implementation (or permitted fallback), output-file layout and retry
  behavior, and the practical access protection needed by the recording service.
- Implement the accepted code-controlled reservation/allocation approach and connect
  approval to recording. A user acknowledgment is not administrator approval; pending
  or refused requests must not be recorded, and approval alone does not prove a write.
- Retain startup readiness, static reuse/seeding, SQLite reset, three-option suggestions,
  and their offline tests when implementing subsequent workflow stages.
- For the later selected hold flow, agree on request timing, interval-based eligibility,
  administrator refusal handling, and on-demand validation/cleanup. A detailed
  reservation-ledger architecture is not a prerequisite for startup and suggestions.
- Finish integration/load tests and assignment documentation. Do not create a backlog
  of hypothetical production UI, tenancy, scaling, observability, or infrastructure work.

## Maintaining documentation

- This file applies repository-wide unless a more specific AGENTS.md applies.
- Keep scope, accepted choices, current status, and real undecided work here. Replace
  stale information; do not append implementation history or speculative roadmaps.
- Keep PROJECT_REQUIREMENTS.md unchanged when documenting implementation choices.
  Record a user-approved scope change separately and explicitly.
- CLAUDE.md imports @AGENTS.md; do not duplicate project context there.
- Setup, commands, and behavior: README.md. Implemented data fields: PARKING_DATA_MODEL.md.
  Code style: pyproject.toml. Retain the MIT LICENSE and existing copyright notices.
