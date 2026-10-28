# Agent Guidelines

## Scope and repository state
- This file applies to the entire repository unless a more specific AGENTS.md applies.
- Python >=3.12 project with LangChain/LangGraph dependencies, shared constants,
  a user-facing system prompt, and a class-owned, import-safe conversational graph.
  All four stages remain incomplete; prompt instructions do not implement enforcement.
- Two-agent, graph-controlled coordination is accepted; detailed workflow topology remains open.
- Verify the current scaffold before implementation; proposals are not accepted decisions.
- Primary entry point: LangGraph CLI + LangSmith Studio (Chat/Graph modes), not a terminal loop.
  Studio is for development/internal testing; public UI and production access controls remain open.
- `ParkingSpaceReservationAgent` in `parking_agent.py` owns immutable `AgentConfig`,
  shared clients/helpers, and deterministic graph construction. `app.py` exports its graph as `agent`.
  The user-facing model-call node is named `parking_assitant` (user-selected spelling),
  distinguishing it from the future administrator-communication agent.
  The default client uses the existing AzureChatOpenAI/DIAL prototype configuration;
  `AZURE_OPENAI_DEPLOYMENT` can override it. Final provider/model selection remains open.
- Conversation/request state belongs to graph threads, never shared mutable instance attributes.
  Agent Server supplies checkpoint persistence; the export does not install a custom checkpointer.
  Models/checkpointers can be injected for tests. Imports do not load `.env`, initialize clients,
  or create databases; CLI loads `.env` and clients initialize lazily once per instance.
  Expected provider/configuration failures are sanitized; unexpected programming errors propagate.
  Traces/checkpoints still need protection. `ParkingState` uses a dataclass and `add_messages`
  to preserve the messages API/Studio schema while avoiding ty's TypedDict protocol-bound error.
- `storage/` implements local persistent Chroma static retrieval and unseeded SQLite dynamic
  facts, owned by the agent as `static_store`/`dynamic_store`, configured by frozen
  `AgentConfig.storage`/`StorageConfig`. Embeddings/stores can be injected. Databases initialize
  on first storage operation, not chat/import; embedding clients initialize on actual embedding.
  The graph does not call storage yet. Paths default to project-root `.data/` (git-ignored).
- `storage/parking_spaces.json` replaces the incorrect guide corpus with eight mock
  individually identified spaces for `demo-central`. `ParkingSpace`/`SpaceFilters` in
  `storage/spaces.py` validate static inventory and allowlisted exact metadata constraints
  before semantic ranking. Documents use `{location_id}:space:{space_id}` IDs and
   generated descriptions. Fields are only space_id/location_id/label/description/source;
   filters support only exact space_id plus mandatory location/type/schema scope.
   Every space fits any vehicle by project assumption; no fit/category/feature modeling.
   The default collection is `parking_spaces_v2` with schema version 2; existing `.env`/
   databases are unchanged. Re-seed into the new collection; do not overwrite old-schema
   IDs in old collections (Chroma merges metadata). Queries exclude old schema/guide records.
- `python -m storage.seed` validates/upserts reviewed structured spaces; `--dry-run` is
  offline, `--file` supports reviewed inventories. No automatic seeding or `.env` loading;
  use `uv run --env-file .env python -m storage.seed`. Updates do not delete omitted IDs
   or enforce approval. `PARKING_DATA_MODEL.md` defines the minimal fields and future
   booking responsibilities; reservation storage/services are not implemented.

## Persistent project context
- Keep project requirements, accepted decisions with rationale, current status, and
  unresolved work here. CLAUDE.md imports @AGENTS.md; do not duplicate this context.
- After each implemented feature, review and update AGENTS.md as needed; prefer simplifying
  and shortening it while preserving requirements and relevant decisions. Also update it
  after meaningful decisions; replace stale context and remove resolved questions, not append history.
- Keep this file simple, concise, and project-specific: no general knowledge, common-sense
  advice, or duplicated guidance.
  Formatter/linter configuration should define code style.
- Use the requirements below as the baseline; ask only about unresolved or new choices.

### Outstanding design work
- The four project stages are the main and only current goal.
- Treat the required car_number as a user-provided vehicle identifier, including for
  vehicles without plates. Do not require vehicle type, measurements, or plate format.
- Chroma was selected for a local, serverless persistent static space index; SQLite for dynamic facts
  without another dependency. Static LangChain documents use validated inventory metadata
  and generated descriptions.
  SQLite stores working_hours/prices/availability as JSON snapshots keyed by location/kind,
  with UTC write timestamps and per-operation transactional connections. This is not a
  reservation ledger or availability enforcement. Domain schemas and refresh rules remain open.
- The lazy Azure/DIAL embedding client defaults to the existing embedding constants with env
  overrides; empty searches do not embed. No tokenizer downloads; ingestion must later chunk
  input to model limits. Model changes require a new/reindexed collection. Telemetry is disabled.
- Trusted Python ingestion only: no LLM write tools, chat ingestion, or personal reservation data
  in either public-fact store. Metadata validation/filtering is not PII detection or authorization.
  Graph retrieval/tool wiring, guardrails, access controls, safe logging, and evaluation remain open.
- Retrieval must be controlled deterministically by Python/LangGraph code and use Chroma
  for static evidence, not depend on an LLM choosing to call a retrieval tool. Each turn supplies
  relevant records/fields as current model context; dynamic operational facts supplement static
  evidence, not write reservation data into Chroma. Query construction and context budgets remain open.
- Separate storage initialization, reviewed seeding/ingestion, readiness validation, and refresh.
  Bootstrap/startup wiring remains open. The user wants static updates gated by administrator
  approval; clarify knowledge-change approval versus reservation approval before implementing writes.
- Availability may change while administrator approval is pending. Decide whether to
  hold capacity or recheck availability before committing a reservation. The user is considering
  pending-request holds to avoid offering the same capacity twice; hold expiry/release rules and
  an authoritative reservation ledger remain undefined, not implemented by public JSON snapshots.
- Human-approval workflows can replay operations when resumed; notifications and
  recording should resist duplicates. Detailed orchestration remains undecided.

### Accepted responsibility boundaries
- Parking-slot allocation, authoritative reservation state, approval enforcement,
  and record formatting/writing belong in Python code, not LLM decisions.
- Allocation may be exposed as an agent tool; tool wiring and reservation storage remain open.
- Agents collect/clarify details and coordinate administrator communication; the
  human administrator alone approves. The recording service owns the file format,
  so the user-facing system prompt need not specify serialization details.
- Use a user-facing parking agent and an administrator-communication agent,
  coordinated by deterministic LangGraph/Python workflow code, not a third orchestrator
  LLM. This keeps the predictable approval lifecycle explicit without another model call.
- The main agent may initiate submission through a tool; code validates readiness,
  dispatches the request, waits for trusted approval/refusal, and controls allocation/recording.

## Project requirements
- Goal: an intelligent parking chatbot with human-in-the-loop reservation confirmation.
- Required language/frameworks: Python, LangChain, and LangGraph.
- Required architecture: Retrieval-Augmented Generation (RAG) with a vector database.
- Recommended vector databases: Milvus, Pinecone, or Weaviate; local Chroma is selected
  for the current prototype (the recommendations are not mandatory).
- Protect sensitive/private information, including any stored in the vector database.
- Evaluate performance, including request latency and information retrieval accuracy.

### Stage 1: RAG system and chatbot
- Implement the basic RAG chatbot architecture and integrate vector storage.
- Answer questions about general parking information, working hours, prices,
  parking-space availability, and location.
- Interactively collect first name, surname, car number, and reservation period.
- Add guardrails and filtering to prevent sensitive-data exposure; pretrained NLP
  text-analysis models are an example approach, not a mandated dependency.
- Evaluate RAG performance and response accuracy, e.g. Recall@K and Precision.
- Outcome: an interactive information chatbot, data protection, and an evaluation report.
- Optional improvement: separate static and dynamic source data. Store static facts
  (general information, parking details, location, booking process) in vector storage;
  store dynamic facts (availability, working hours, prices) in SQL storage.

### Stage 2: Human-in-the-loop agent
- Create a second agent using LangChain to communicate with the administrator.
- After the first agent collects reservation details, escalate the request to the human.
- Generate/send approval requests, receive explicit confirmation or refusal, and
  maintain communication between the first and second agents.
- Email, messenger, and REST API are example channels; the choice remains open.
- Outcome: an automated system that contacts an administrator for reservation approval.
- The second agent coordinates the human decision; it must not approve autonomously.

### Stage 3: Process confirmed reservations using an MCP server
- Integrate an open-source MCP server capable of writing data to a file, or develop
  a simple Python/FastAPI MCP server to process confirmed reservations.
- If MCP cannot be implemented, tool/function calling to write a file is allowed;
  document this fallback rather than describing it as an MCP implementation.
- Only after human administrator approval, write reservation details to a text file.
- Required file-entry format: `Name | Car Number | Reservation Period | Approval Time`.
- Collect both first name and surname; define how they compose the `Name` field.
- Ensure secure, reliable service and resistance to unauthorized access.
- Outcome: a recording service integrated with both agents that saves approved data.

### Stage 4: Orchestrate all components via LangGraph
- Implement the complete workflow using LangGraph: RAG/user interaction → human
  administrator approval via the second agent → MCP processing and data recording.
- Example graph nodes: user interaction, administrator approval, and data recording;
  these responsibilities are required, but exact graph topology is not prescribed.
- Test the complete workflow and integration of all stages for stable operation.
- Load-test interactive chatbot dialogue, administrator confirmation functionality,
  and MCP recording/storage separately, as well as the integrated pipeline.
- Document architecture, agent/server logic, usage, setup, and deployment guidelines.
- Outcome: a unified, fully integrated system with a stable end-to-end pipeline.

### Submission and additional-credit artifacts
- For each stage, provide a GitHub or EPAM GitLab repository link in the answer field.
- Additional credit: a PowerPoint presentation explaining the solution with screenshots.
- Additional credit: README documentation covering setup, usage, and project structure.
- Additional credit: automated pytest or unittest tests, at least two tests per module.
- Additional credit: CI/CD automation and/or infrastructure as code, e.g. Terraform.
- Poor-quality, impractically basic code or critical errors may reduce the grade.

### Implementation boundaries and open decisions
- A request acknowledgment is not administrator approval; approval is not a successful write.
- Pending/refused requests must not be recorded as approved reservations.
- Model-generated output is not authoritative approval or reservation data.
- Final provider/model, communication channel, MCP implementation,
  authentication, detailed states, timezone semantics, and file location remain undecided.
- Document new decisions without turning optional specification examples into requirements.
- Retain LICENSE (MIT) and existing copyright notices.

## Development references
- Setup, Studio usage, and verification commands: README.md.
- Lint/format and type-checker configuration: pyproject.toml.
