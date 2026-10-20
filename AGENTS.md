# Agent Guidelines

## Scope and repository state
- This file applies to the entire repository unless a more specific AGENTS.md applies.
- Python >=3.12 scaffold with pyproject.toml and uv.lock; no dependencies,
  application modules, or tests. All four stages remain incomplete.
- No architecture, storage design, or application toolchain is approved/configured.
- Verify the current scaffold before implementation; proposals are not accepted decisions.

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
- Before finalizing data storage, resolve single versus multiple parking locations and
  interchangeable capacity versus individually bookable spaces.
- Decide whether to separate static RAG knowledge from live structured data; this is
  an optional design choice, not an accepted architecture.
- Consider keeping reservation personal data out of the general-purpose RAG index;
  protect private data through access controls and safe logging, not prompts alone.
- Availability may change while administrator approval is pending. Decide whether to
  hold capacity or recheck availability before committing a reservation.
- Human-approval workflows can replay operations when resumed; notifications and
  recording should resist duplicates. Exact orchestration remains undecided.
- No implementation task is currently approved; follow the user's next request.

## Project requirements
- Goal: an intelligent parking chatbot with human-in-the-loop reservation confirmation.
- Required language/frameworks: Python, LangChain, and LangGraph.
- Required architecture: Retrieval-Augmented Generation (RAG) with a vector database.
- Recommended vector databases: Milvus, Pinecone, or Weaviate; none is selected yet.
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
- Provider/model, vector database, communication channel, SQL use, MCP implementation,
  authentication, detailed states, timezone semantics, and file location remain undecided.
- Document new decisions without turning optional specification examples into requirements.
- Retain LICENSE (MIT) and existing copyright notices.

## Build, lint, and test commands
- No application entry point, build, formatter, linter, type checker, or test runner is
  configured. Do not report these checks as passed or run guessed commands.
- When tooling is introduced, record verified commands here and setup in README.md.
  Automated unit tests remain additional credit; Stage 4 workflow/load testing is required.
