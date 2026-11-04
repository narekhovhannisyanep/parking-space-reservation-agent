# Parking data model

This reference describes the implemented schemas, state, lifecycle, and data boundaries,
not additional assignment requirements. The fixed assignment baseline is
[PROJECT_REQUIREMENTS.md](PROJECT_REQUIREMENTS.md); durable scope and contributor
instructions are in [AGENTS.md](AGENTS.md).

## Assumptions

- Spaces have individual IDs; allocation belongs in Python code.
- Every space fits any vehicle. No categories, dimensions, clearance checks, or
  facility-feature matching are needed.
- `car_number` means the user-provided vehicle identifier, including for vehicles
  without plates. Do not require a plate format or invent an identifier.
- Public parking information and private reservation details remain separate.

## Static inventory

`storage/parking_spaces.json` contains 20 mock spaces at fictional `demo-central`.
Each record requires exactly these fields:

| Field | Purpose |
| --- | --- |
| `space_id` | Space identity within a location |
| `location_id` | Location scope |
| `label` | Human-visible space marking |
| `description` | Realistic mock location, route, practical benefits and trade-offs |
| `source` | Source attribution |
| `hourly_price_cents` | Required integer USD cents/hour, 100–2000 inclusive; never a boolean or inferred default |

Current validation requires trimmed, nonblank strings. IDs contain 1–64
alphanumeric/dash/underscore characters and start alphanumerically. Description is
limited to 400 characters; label/source to 200. Unknown fields are rejected; this
does not itself detect sensitive content in allowed text fields.

Ingestion additionally rejects detected PII/secrets before embedding; selected
existing evidence is checked before graph/model use. Local privacy checks supplement
schema validation and operator review, not prove source safety.

`ParkingSpace` generates LangChain documents with IDs `{location_id}:space:{space_id}`.
Metadata contains the fields except description, plus `record_type=parking_space`
and integer `schema_version=3`. Prices are metadata only, not embedded document text;
currency is fixed to USD. `SpaceFilters` supports optional exact `space_id` and a
validated `HourlyBudget` with numeric `$lt`/`$lte`/`$gt`/`$gte` clauses;
search always retains location/type/schema scope before semantic ranking.

Chroma's default collection is `parking_spaces_v3`. Use a fresh schema-3 collection
when migrating from an older schema: Chroma merges metadata on upsert, and same-ID
cross-schema upserts are refused. Preserve older collections and follow README's
migration and seeding instructions; do not erase or reindex populated data automatically.
The fixture is intentionally limited public space/location evidence, not a complete
facility knowledge base. Hourly prices come solely from Chroma metadata. Missing hours
remain unknown until provided through Python CRUD. Descriptions are fictional route
evidence, not certified accessible bays, vehicle-fit flags, or hard feature constraints.
See RAG_EVALUATION.md for measured information coverage and limits.

At server startup, missing/empty configured collections are immediately seeded from
the mock fixture. Populated collections are reused unchanged, without freshness checks.

## Dynamic information

SQLite supplies `working_hours` and `availability` as JSON snapshots keyed
by `(location_id, kind)`, with UTC `updated_at` write timestamps. CRUD operations
are transactional. Values must be JSON-compatible with string keys and finite numbers.
Dynamic storage is recreated empty at each application-server startup; writes require
explicit Python calls. This reset does not happen for new conversation threads.

Information turns read these snapshots and instruct the model to use them for
operational facts rather than static prose or earlier chat. Missing/null/omitted
hours mean unknown. Write timestamps do not prove freshness, and snapshots do not
allocate spaces or establish availability for a requested booking period.
Availability suggestions use a location-wide boolean snapshot or a space-ID/boolean
map. `False` blocks the applicable space(s); absent snapshots or absent map IDs leave
the static-available-by-default assumption in effect. Null, counts, string statuses,
and other unsupported shapes fail selection rather than silently offering spaces.
This is a public snapshot contract, separate from the private Stage 2 ledger below.
The same SQLite startup reset recreates both tables empty.

The legacy Python CRUD kind `prices` remains accepted for API compatibility, but is
never read into RAG evidence or used for filtering/display. Do not populate it: prices
have one authoritative source, the Chroma inventory. Price-only questions read
scoped metadata to compute a minimum/maximum summary without embeddings or full prose.

Python reads inventory IDs and applies dynamic eligibility before semantic ranking.
Chroma embeds the fixed public-topic query derived from the latest user message and
returns up to three eligible similarity matches using a native result limit. Optional
hourly price constraints apply to both the matching-record count and semantic query,
before top-K ranking. No hard feature flags are modeled, and the whole inventory is not
randomly sampled. Only retrieved descriptions/rates appear in `availability_options`.
Non-offer retrieval uses `static_spaces` for directions/identity. Offered `space_id`
values live in the thread's internal `offered_space_ids` list; suggestions never write
or hold spaces.
`offer_candidate_ids` preserves the current browse choices through collection and
duration changes; `offer_query` contains only code-owned public-topic terms. Confirmation
automatically rechecks candidates, keeping eligible IDs/order and filling only vacancies
with native ranked eligible results. `/options` optionally rechecks the same choices;
it is not required before submission. Removed IDs appear in `no_longer_eligible` evidence
and are explained, not presented as options. No raw preference text or personal fields
are persisted in this public-topic query.
`offer_round_ids` accumulates every ID offered since the latest fresh request; an
"other options" follow-up excludes them before ranking. When set, the evidence also
carries `previous_offers_excluded` (a count, not IDs).
The server also prepares bounded baseline evidence before interaction, without adding
the entire static inventory to prompts.

## Optional hourly budget

`ReservationExtraction.budget_phrase` is a separate optional verbatim phrase in the
existing structured extraction call, allowed on information and reservation turns.
It never becomes a fifth required field or enters the private reservation/MCP schema.
`pricing.py` independently parses one bounded English USD/hour upper limit or inclusive
range, validates model grounding, and preserves strict versus inclusive upper limits.
Recognizable unsupported, conflicting, total-stay, or unit-ambiguous price cues fail
closed; structured output alone does not prove semantic correctness. Decimal amounts
must represent whole cents, number words zero through twenty are supported, and user
limits are not clamped to the inventory range. Amounts above $10,000,000/hour are
rejected as unsupported rather than rounded. This is not universal language understanding.

Thread `budget` stores only normalized cent bounds/inclusivity; `budget_pending` blocks
new options and submission until clarification or `/clear-budget`. Preferences persist
through `/options` and other-options turns; changes invalidate offers/round IDs but not
confirmed personal details. `/new` and `/cancel` reset the preference. Pending submitted
requests cannot be edited. Submission rereads the selected rate when a budget is active;
static prices are reviewed mock data, not a concurrently managed billing system or a
locked quote. The existing atomic SQLite availability/overlap check remains separate.
Code appends every offered ID, description, and USD/hour rate to guarded information
and collection replies. General model explanation/grounding remains prompt-guided;
the authoritative option entries, not arbitrary model prose, enforce the selected list.

## Reservation details

Required user fields are first name, surname, `car_number`, and reservation period.
The requested period is represented as positive `duration_hours`. Users supply only
how many hours, never a date/time or timezone.
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
Extraction also allows `rejections` (field plus complete verbatim `phrase`) and optional
`disagreement_phrase`, never approval/submission authority. Python independently checks
evidence and common bounded English negative clauses. A named rejection clears its
field; a positive same-message replacement may fill it. Negated occurrences and space
references/fragments cannot populate the vehicle field, while explicitly labeled arbitrary
vehicle identifiers remain allowed. Vague disagreement sets session `correction_pending`,
clears request/confirmation/summary and requires a concrete correction or clear before
another confirmation. These are private draft state, not SQLite/MCP data fields or
universal semantic correctness guarantees.

`ready` means user-confirmed details, **not submission or administrator approval**.
Personal fields are not written to Chroma or public SQLite snapshots. They are
present in model inputs and private graph checkpoints; Studio can display them.
Necessary details reach extraction, while detected unrelated PII is filtered and
updates are grounded in sanitized text. Extraction permits PERSON spans, so unrelated
names in the latest message may also reach the provider. General-answer history/model outputs are
sanitized. Thread `privacy_values` retains collected names/vehicle identifiers to
filter historical mentions after corrections/cancellation/new drafts. This is not
deletion or access control. See README and GUARDRAIL_EVALUATION.md for local detection,
streaming, provider, checkpoint, and language limitations.

## Private reservation ledger and administrator approval

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

The user confirms details, receives automatically rechecked options, and submits one
offered ID with `/submit SPACE_ID`. Offers preview a now-based UTC window and bind to
the validated duration, not a fixed start. Duration changes invalidate confirmation and
eligibility binding, not browse candidates; fresh confirmation automatically rechecks
them. Delays alone do not.
Pending/approved rows block overlaps using `[start, end)`;
adjacent periods can share a space, including within the same thread. Unknown-period
suggestions use current observations and exclude only holds active now. Before ranking,
Python reads only blocked space IDs from the ledger, never private request fields.
RAG evidence includes `eligibility_scope` and, when applicable, `reservation_period_utc`.
Suggestions never create holds; submission rechecks static inventory, public overrides,
the actual server-generated submission period, and overlapping holds. Conflicts refresh
the semantic options. Time before submission does not consume the requested duration;
waiting for administrator review after submission does count and never shifts the end.

Internal graph state adds `offer_duration_hours`, `refresh_options`, `revalidate_options`, `submission`,
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
releases the hold. Ended periods stop blocking on-demand queries without deleting
records or running timers. No editing, cancellation, or new request is allowed while
waiting; `/new` after the decision starts a separate one-space request. Earlier approved
requests remain authoritative and block
their overlapping periods. Startup deliberately clears all ledger rows; old graph
checkpoints cannot recreate a missing submitted request on review.

Studio is the selected **trusted local** review channel, with no out-of-band notification
or separate administrator authentication. Thread/request matching correlates decisions,
not proves human identity or secures development APIs. Necessary submitted fields reach
the configured administrator-composition model and remain visible in private SQLite,
Studio, checkpoints, and debug data. Public model copies/streams retain the privacy
boundaries described in README; raw administrator model tokens use `nostream`.

## Approved file recording

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

Protection is a trusted-local process/OS boundary, not separate human authentication
or a sandbox. See README for primitives, failure behavior and limitations, and
STAGE4_VALIDATION.md for full-system validation evidence and its limits.
