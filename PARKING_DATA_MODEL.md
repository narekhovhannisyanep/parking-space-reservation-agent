# Parking data model

This describes implemented data and accepted choices, not additional assignment
requirements. The fixed baseline is [PROJECT_REQUIREMENTS.md](PROJECT_REQUIREMENTS.md);
scope and decisions are in [AGENTS.md](AGENTS.md). Implement the four stages without
adding production-grade parking management or speculative future-proofing.

## Accepted simplifying assumptions

- Spaces have individual IDs; allocation belongs in Python code.
- Every space fits any vehicle. No categories, dimensions, clearance checks, or
  facility-feature matching are needed.
- `car_number` means the user-provided vehicle identifier, including for vehicles
  without plates. Do not require a plate format or invent an identifier.
- Public parking information and private reservation details remain separate.

## Static inventory (implemented)

`storage/parking_spaces.json` contains 20 mock spaces at fictional `demo-central`.
Each record requires exactly these fields:

| Field | Purpose |
| --- | --- |
| `space_id` | Space identity within a location |
| `location_id` | Location scope |
| `label` | Human-visible space marking |
| `description` | Short public directions/information |
| `source` | Source attribution |

Current validation requires trimmed, nonblank strings. IDs contain 1–64
alphanumeric/dash/underscore characters and start alphanumerically. Description is
limited to 400 characters; label/source to 200. Unknown fields are rejected; this
does not itself detect sensitive content in allowed text fields.

Ingestion additionally rejects detected PII/secrets before embedding; selected
existing evidence is checked before graph/model use. Local privacy checks supplement
schema validation and operator review, not prove source safety.

`ParkingSpace` generates LangChain documents with IDs `{location_id}:space:{space_id}`.
Metadata contains the fields except description, plus `record_type=parking_space`
and integer `schema_version=2`. `SpaceFilters` supports optional exact `space_id`;
search always retains location/type/schema scope before semantic ranking.

Chroma's default collection is `parking_spaces_v2`. Use a fresh collection when
moving from an older schema: Chroma merges metadata on upsert. See README for seeding.
The fixture is intentionally limited public space/location evidence, not a complete
facility knowledge base. Hours/prices remain unknown until provided through public
Python CRUD; see RAG_EVALUATION.md for measured information coverage and limits.

At server startup, missing/empty configured collections are immediately seeded from
the approved mock fixture. Populated collections are reused unchanged. Static-data
change/approval workflows are outside the agreed scope.

## Dynamic information (storage and retrieval implemented)

SQLite stores `working_hours`, `prices`, and `availability` as JSON snapshots keyed
by `(location_id, kind)`, with UTC `updated_at` write timestamps. CRUD operations
are transactional. Values must be JSON-compatible with string keys and finite numbers.
Dynamic storage is recreated empty at each application-server startup; writes require
explicit Python calls. This reset does not happen for new conversation threads.

Information turns read these snapshots and instruct the model to use them for
operational facts rather than static prose or earlier chat. Missing/null/omitted
hours/prices mean unknown. Write timestamps do not prove freshness, and snapshots do not
allocate spaces or establish availability for a requested booking period.
Availability suggestions use a location-wide boolean snapshot or a space-ID/boolean
map. `False` blocks the applicable space(s); absent snapshots or absent map IDs leave
the static-available-by-default assumption in effect. Null, counts, string statuses,
and other unsupported shapes fail selection rather than silently offering spaces.
This is a public snapshot contract, separate from the private Stage 2 ledger below.
The same SQLite startup reset recreates both tables empty.

Python reads inventory IDs and applies dynamic eligibility before semantic ranking.
Chroma embeds the fixed public-topic query derived from the latest user message and
returns up to three eligible similarity matches using a native result limit. No space
features are modeled, and the whole inventory is not randomly sampled. Only retrieved
descriptions appear in `availability_options`. Non-offer retrieval
uses `static_spaces` for directions/identity. Offered `space_id` values live in the
thread's internal `offered_space_ids` list; suggestions never write or hold spaces.
`offer_round_ids` accumulates every ID offered since the latest fresh request; an
"other options" follow-up excludes them before ranking. When set, the evidence also
carries `previous_offers_excluded` (a count, not IDs).
The server also prepares bounded baseline evidence before interaction, without adding
the entire static inventory to prompts. No occupancy-feed/refresh-service platform is needed.

## Reservation details (collection implemented)

Required user fields are first name, surname, `car_number`, and reservation period.
The user-approved simplification represents the requested period as positive
`duration_hours`. Users supply only how many hours, never a date/time or timezone.
Python derives start/end from the application server's UTC clock at submission.

`reservation_collection.py` uses field proposals, a draft, a thread-scoped session,
and `ValidatedReservationRequest` (identity, names, vehicle identifier, and numeric
duration only). Current code accepts allowlisted verbatim user values, validates them,
renders a revisioned summary, and asks for confirmation. The duration is validated by
Python, not inferred from dates or units. No provisional start/end is confirmed.
States are `idle`, `collecting`, `awaiting_confirmation`, `ready`, and `cancelled`.
Corrections/clears invalidate confirmation, but clock progress does not. See README
for commands and supported hour expressions. Old scheduled draft checkpoints retain
personal fields but discard schedule/confirmation and require hours plus fresh confirmation.

`ready` means user-confirmed details, **not submission or administrator approval**.
Personal fields are not written to Chroma or public SQLite snapshots. They are
present in model inputs and private graph checkpoints; Studio can display them.
Necessary details reach extraction, while detected unrelated PII is filtered and
updates are grounded in sanitized text. General-answer history/model outputs are
sanitized. Thread `privacy_values` retains collected names/vehicle identifiers to
filter historical mentions after corrections/cancellation/new drafts. This is not
deletion or access control. See README and GUARDRAIL_EVALUATION.md for local detection,
streaming, provider, checkpoint, and language limitations.

## Private reservation ledger and administrator approval (Stage 2 implemented)

`storage/reservations.py` uses a separate `parking_reservations` table in the configured
SQLite database. Public fact CRUD and RAG evidence never return these records.

| Field | Purpose |
| --- | --- |
| `request_id` | Primary key `draft_id:revision`; idempotent submission identity |
| `thread_id` | Owning graph thread; application lookups/decisions are thread-scoped |
| `location_id`, `space_id` | Scoped selected inventory space |
| `request_json` | Private canonical `SubmittedReservationRequest`: names, vehicle identifier, duration hours, server-generated UTC period and draft identity |
| `start_at`, `end_at` | Fixed-microsecond UTC timestamps for indexed half-open interval checks |
| `status` | `pending`, `approved`, `refused`, or `elapsed` |
| `submitted_at` | UTC hold creation time |
| `decision_at` | UTC completed review time; null while pending |
| `approval_time` | UTC human approval time only; null for pending/refused/elapsed |

The table constrains statuses, ordered intervals, and status/timestamp consistency.
Submission acquires the write lock, captures current UTC time as `start_at`/`submitted_at`,
calculates `end_at = start_at + duration_hours`, rechecks snapshots and overlap, then
inserts the hold in the same short transaction. No LLM or human review happens
inside a database transaction. Existing identity retries must match thread, space,
and collected details exactly; retries preserve the original period even as the clock
advances. Completed decisions cannot be reversed.

The user confirms details, asks for `/options`, and submits one offered ID with
`/submit SPACE_ID`. Offers preview a now-based UTC window and bind to the validated
duration, not a fixed start. Duration changes invalidate them; delays alone do not.
Pending/approved rows block overlaps using `[start, end)`;
adjacent periods can share a space, including within the same thread. Unknown-period
suggestions use current observations and exclude only holds active now. Before ranking,
Python reads only blocked space IDs from the ledger, never private request fields.
RAG evidence includes `eligibility_scope` and, when applicable, `reservation_period_utc`.
Suggestions never create holds; submission rechecks static inventory, public overrides,
the actual server-generated submission period, and overlapping holds. Conflicts refresh
the semantic options. Time before submission does not consume the requested duration;
waiting for administrator review after submission does count and never shifts the end.

Internal graph state adds `offer_duration_hours`, `refresh_options`, `submission`,
`administrator_payload`, and `review_retry`. `submission` holds only request identity,
space, and status; SQLite supplies canonical details on review/replay. The second
LangChain agent generates bounded neutral review wording from the submitted request.
Python builds the exact private review payload and displays it through a typed Studio
interrupt; the human returns only `{decision}` with `approve` or `refuse`. No default
decision or manual request-ID entry is provided. Python uses the checkpointed submission
and current thread to load SQLite and checks the internal review-payload ID matches that
record. Invalid human responses open a fresh interrupt without changing SQLite; an
internal identity mismatch fails closed before showing/applying a decision. The first
agent reports the code-controlled outcome, not model-generated authority.

Refusal releases the hold; approval before end retains it and its approval time, even
though start has passed. Approval at/after end becomes `elapsed`, never approved, and
releases the hold. Ended periods
stop blocking on-demand queries without deleting records or running timers. No editing,
cancellation, or new request is allowed while waiting; `/new` after the decision starts
a separate one-space request. Earlier approved requests remain authoritative and block
their overlapping periods. Startup deliberately clears all ledger rows; old graph
checkpoints cannot recreate a missing submitted request on review.

Studio is the selected **trusted local** review channel, with no out-of-band notification
or separate administrator authentication. Thread/request matching correlates decisions,
not proves human identity or secures development APIs. Necessary submitted fields reach
the configured administrator-composition model and remain visible in private SQLite,
Studio, checkpoints, and debug data. Public model copies/streams retain the privacy
boundaries described in README; raw administrator model tokens use `nostream`.

## Approved file recording (Stage 3 implemented)

Only human-approved reservations may be written as:

`Name | Car Number | Reservation Period | Approval Time`

`Name` is first name + one space + surname. The period is the original UTC ISO start
`to` UTC ISO end; approval is the original human approval instant rendered in UTC.
Each approved request has one UTF-8 `.txt` file with one line, named by SHA-256 of
its internal request ID. No space/request ID or extra column is included in the entry.
Pending/refused/elapsed requests are not approved records. Approval is not proof that
recording succeeded; recording after end is allowed for a previously valid approval.

SQLite is the authoritative operational reservation state. The Stage 3 file is an
approved-only output, not another source to synchronize/read for allocation. A failed
file write retains approval and allocation in SQLite and must be reported explicitly.
The dedicated disposable directory is `approved-reservations/` beside configured
SQLite (`.data/approved-reservations/` by default), dropped/recreated only at server
startup. Path/symlink/overlap checks protect the parent and persistent Chroma. Output
directories are owner-only `0700`; SQLite and generated files are `0600`.

The official Python SDK serves a private stdio MCP tool taking only internal
`request_id`/`thread_id`. The server reads existing SQLite in read-only mode, checks
approved status and canonical consistency, and owns validation/formatting/publication.
It accepts no body, approval flag, or arbitrary output path. Atomic no-overwrite
publication and matching-content replay provide idempotent file effects, not exactly-once
message delivery. Files are checked only for recording replay, never allocation.

Internal thread state adds `recording` (a JSON `RecordingResult`) and
`recording_requested`. The result has `status` (`recorded`, `failed`, `unconfirmed`),
`request_id`, optional successful `filename`, optional safe `error_code`, `retryable`,
and `attempts` (1–2). It contains no personal fields or raw protocol/exception data.
These are recording outcomes, not new ledger statuses/fields. `/new` clears them.

The client makes one initial attempt plus at most one transient-failure retry;
`/retry-recording` repeats only recording for the current approved failed/unconfirmed
request before `/new`. Invalid/permanent failures are not automatically retried.
Timeouts/post-publication failure can leave a file without an acknowledged success;
report unconfirmed output honestly and rely on idempotent retry. Approval reversal,
background retries, file-to-SQLite synchronization and cross-restart recovery are out
of scope. Missing ledger rows after reset cannot be reconstructed from checkpoints/files.

Protection is the selected trusted-local process/OS boundary, not separate human
authentication or a sandbox. See README for primitives, failure behavior and limitations.
Actual stdio and Agent Server integration, replay, concurrent writing, failure and
cleanup tests are implemented. STAGE4_VALIDATION.md records the completed full-system
local load and two real-provider pipeline smoke checks. Load uses synthetic data/
injected models; live smoke does not establish browser accuracy, live-provider load
capacity, or complete confidentiality. Stage 4 adds no operational data fields or
file-to-ledger synchronization.
