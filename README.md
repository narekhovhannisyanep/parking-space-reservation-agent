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
- Future vector storage/retrieval helpers for static facts and SQL services for
  dynamic data will be initialized and owned by this class. The split is the chosen
  design direction; engines, source data, and schema are not implemented or finalized.
  Chroma and SQLite are examples under consideration, not installed dependencies.

Importing `app` creates the instance and graph but does not read `.env`, initialize
an API client, create databases, or make network calls. Tests can inject a model
through `ParkingSpaceReservationAgent(llm_client=...)` and an optional checkpointer
for standalone use. The server export intentionally has no custom checkpointer:
Agent Server supplies persistence. Without the server or an injected checkpointer,
direct callers must supply conversation history themselves.

This remains a conversational prototype: RAG, live availability, administrator
approval, allocation, and recording are **not implemented**. It cannot complete reservations.
Inputs are sent to the configured model provider. Keep secrets out of chat and leave
LangSmith tracing disabled unless personal-data handling and access controls are configured.
The template disables tracing; an existing `.env` is not changed automatically.
Studio can still connect with tracing disabled. This does not prevent sending inputs
to your model provider or saving them in local checkpoints. If you enable tracing,
prompts, responses, and personal data may also be stored in LangSmith. Sanitized
graph errors do not replace access controls, checkpoint protection, or trace redaction.
Only expected OpenAI API and client-configuration errors are sanitized; programming
errors propagate so they can be diagnosed rather than silently masked.

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
They do not make API calls or evaluate live model behavior, retrieval accuracy, or security enforcement.

Type, source lint, and formatting checks can run through uv's temporary tool environments
without changing the project dependencies:

```bash
uv tool run ty check
uv tool run ruff check app.py parking_agent.py commons tests
uv tool run ruff format --check app.py parking_agent.py commons tests
```

Ruff enables `BLE001` to prevent broad exception catches from returning.

References: [local server](https://docs.langchain.com/langsmith/local-server),
[Studio setup](https://docs.langchain.com/langsmith/quick-start-studio), and
[server-managed persistence](https://docs.langchain.com/oss/python/langgraph/persistence).
