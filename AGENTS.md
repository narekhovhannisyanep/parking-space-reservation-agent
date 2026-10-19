# Agent Guidelines

## Scope and repository state
- This file applies to the entire repository unless a more specific AGENTS.md applies.
- The project is currently a scaffold, not an implemented application.
- README.md introduces the project; the requirements checkpoint below adds detail.
- Initially tracked files: README.md, LICENSE, and .gitignore.
- A Python scaffold now includes pyproject.toml and uv.lock; dependencies are empty.
- pyproject.toml requires Python >=3.12; there are no application modules or tests yet.
- No application build, formatter, linter, or type checker is configured.
- Do not present proposed tooling or architecture as an existing convention.
- Reinspect the repository before working; these facts will change as code is added.

## Project requirements checkpoint
Source: the user's technical specification and four stage screenshots, supplied 2026-10-19.
These are implementation requirements, not claims that the capabilities already exist.
All stages are shown with the same Oct 24–Nov 7, 2026 window (14 days).
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

## Existing agent and editor rules
- No preexisting root AGENTS.md was found during inspection.
- No .cursor/rules/ directory or .cursorrules file was found.
- No .github/copilot-instructions.md file was found.
- No additional Cursor or Copilot instructions therefore need to be incorporated.
- If these files appear later, read them before changing the affected code.
- Respect applicable instruction precedence; flag incompatible repository guidance.
- .codemie/ is local agent tooling, not an application source directory.
- Do not inspect or modify .git/ internals as part of application development.

## Build, lint, and test commands
Run repository commands from the repository root.

| Task | Current command/status |
| --- | --- |
| Install dependencies | uv scaffold exists; no application dependencies declared yet. |
| Run the application | Not configured; no application entry point exists. |
| Build/package | Not configured; no build configuration exists. |
| Lint | Not configured; no linter configuration exists. |
| Format/check formatting | Not configured; no formatter configuration exists. |
| Type-check | Not configured; no type-checker configuration exists. |
| Run all tests | Not configured; no test suite or runner exists. |
| Run one test file | Not available until a test runner and tests are added. |
| Run one test case | Not available until the runner's selector is documented. |

Available repository inspection and whitespace checks:
```sh
git status --short --untracked-files=all
git ls-files
git diff --check
git diff --cached --check
git diff -- README.md .gitignore AGENTS.md
```
- These Git commands are not substitutes for application linting or tests.
- Ordinary git diff omits untracked files, including newly created documents.
- For a new AGENTS.md, check it without staging:
  `git diff --no-index --check -- /dev/null AGENTS.md`.
- Do not run guessed npm, pytest, uv, or LangGraph commands in this scaffold.
- LangChain/LangGraph are required by the specification but not yet declared dependencies.
- Report build/lint/test checks as unavailable, not passed, while tooling is absent.

## When introducing a toolchain
- Implement in Python with LangChain and LangGraph as required by the specification.
- Inspect any newly added manifests and CI configuration before choosing commands.
- Use the selected package manager consistently and retain its lockfile if applicable.
- Declare dependencies in project configuration, not only in a local environment.
- Configure formatting, linting, and tests alongside the first implementation.
- Record runtime versions and reproducible setup instructions in README.md.
- Replace unavailable command entries above with verified, copyable commands.
- Include exact commands for one test file and one named test case.
- Explain selectors, working directories, fixtures, and required environment variables.
- Separate fast offline tests from integration tests requiring external services.
- Do not add a framework solely to validate a documentation-only change.

## Code style: status and general defaults
No application source code exists from which to infer established code style.
The following are defaults for new code, not observations of an existing codebase.
- Follow Python idioms and committed tool configuration; do not invent formatter settings.
- Prefer small, focused modules and functions with explicit responsibilities.
- Separate conversation orchestration, reservation logic, and external-service access.
- Keep business rules independent of network clients and model-provider SDKs.
- Avoid speculative abstractions and unrelated refactors.
- Make dependencies and side effects explicit rather than relying on global state.

## Imports and dependencies
- Group standard-library, third-party, and internal imports where idiomatic.
- Let the configured import sorter determine ordering and spacing once one exists.
- Use explicit imports; avoid wildcard imports and unused dependencies.
- Keep module imports free of network requests and other heavyweight side effects.
- Avoid circular dependencies; extract shared domain contracts when necessary.
- Prefer a stable module boundary over importing another module's private helpers.
- Add dependencies only when their value outweighs their maintenance cost.

## Formatting and documentation
- Use UTF-8 text, consistent indentation, and a final newline in new text files.
- Avoid trailing whitespace and mixed tabs/spaces in indentation.
- Let the chosen formatter define line width, quotes, and language-specific layout.
- Do not reformat unrelated files while making a focused change.
- Use descriptive Markdown headings, concise bullets, and fenced command examples.
- Use repository-relative paths in documentation rather than machine-specific paths.
- Explain why a decision exists; avoid comments that merely restate the code.
- Document public interfaces, non-obvious invariants, and operational requirements.

## Types and data contracts
- Use explicit types for public interfaces when supported by the chosen language.
- Model parking spaces, reservation requests, and approval decisions deliberately.
- Distinguish pending requests from confirmed bookings in the data contract.
- Represent absence and failures explicitly rather than with ambiguous sentinel values.
- Validate user input and external responses at system boundaries.
- Avoid unchecked casts, blanket type suppression, and unstructured data where practical.
- Define timezone and interval semantics before implementing date-based availability.
- Never treat model-generated output as authoritative reservation or approval data.

## Naming conventions
- Use domain terms consistently: parking space, reservation, user, administrator.
- Use snake_case for Python functions/modules/variables, PascalCase for classes,
  and UPPER_SNAKE_CASE for constants.
- Use descriptive names; avoid unexplained abbreviations and generic helper names.
- Prefer action-oriented function names and predicate-style boolean names.
- Reserve constant-style naming for genuine constants according to language convention.
- Make names distinguish requested, pending, confirmed, rejected, and cancelled states.

## Error handling and security
- Handle expected validation and availability failures explicitly.
- Catch specific errors at appropriate boundaries; do not silently swallow exceptions.
- Preserve useful diagnostic context without exposing credentials or personal data.
- Present actionable, user-safe messages rather than internal stack traces.
- Bound external-call timeouts and retries; avoid retrying booking writes blindly.
- Design write operations to resist duplicate submissions before adding automatic retries.
- Require authorized administrator actions for confirmation when that flow is implemented.
- Keep secrets out of source, logs, test fixtures, and committed configuration.
- No environment variables are currently documented or required by the scaffold.
- If configuration is added, document placeholders in an example environment file.
- Explicitly ignore real .env files before creating them; .gitignore does not do so yet.

## Testing and completion expectations
- Add focused regression tests with behavior changes once a test harness exists.
- Prefer deterministic tests with mocked model, network, clock, and administrator inputs.
- Test validation failures, unavailable spaces, approval/rejection, and duplicate requests.
- Test timeout/error paths and ensure failed operations cannot imply confirmation.
- Cover date boundaries and concurrent booking attempts if those features are introduced.
- Run the smallest relevant test first, then the relevant suite and configured checks.
- For documentation-only changes now, review accuracy and run the Git whitespace checks.
- Preserve unrelated user changes, including preexisting .gitignore modifications.
- Do not stage, commit, push, or change Git configuration without explicit authorization.
- Summarize changed files, checks actually run, and any verification limitations.
- Keep this guide current when application structure, rules, or commands are introduced.
