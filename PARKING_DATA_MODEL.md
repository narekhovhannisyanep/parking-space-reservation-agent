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
The fixture is space inventory, not complete coverage of all required general
parking and location questions. That information coverage remains Stage 1 work.

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
This is a public snapshot contract, not a reservation/hold ledger schema. Period-based
holds, their statuses, and cleanup will be agreed in the subsequent reservation change.

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
Current code represents the period with start/end and an explicit timezone/offset,
normalizes to UTC, and requires a future start and an end after start.

`reservation_collection.py` uses field proposals, a draft, a thread-scoped session,
and `ValidatedReservationRequest`. Current code accepts allowlisted verbatim user
values, validates them, renders a revisioned summary, and asks for confirmation.
States are `idle`, `collecting`, `awaiting_confirmation`, `ready`, and `cancelled`.
Corrections/clears invalidate confirmation. See README for commands and date formats.

`ready` means user-confirmed details, **not submission or administrator approval**.
Personal fields are not written to Chroma or public SQLite snapshots. They are
present in model inputs and private graph checkpoints; Studio can display them.
Necessary details reach extraction, while detected unrelated PII is filtered and
updates are grounded in sanitized text. General-answer history/model outputs are
sanitized. Thread `privacy_values` retains collected names/vehicle identifiers to
filter historical mentions after corrections/cancellation/new drafts. This is not
deletion or access control. See README and GUARDRAIL_EVALUATION.md for local detection,
streaming, provider, checkpoint, and language limitations.

## Administrator approval and recording (not implemented)

The accepted workflow is collection → human approval/refusal via the second agent
→ code-controlled recording. Allocation/reservation state and file formatting are
Python responsibilities. The simplest implementation that fulfills the assignment
and these accepted choices should be selected when implementing the next stages;
there is no agreed new reservation-ledger schema in this document.

Only human-approved reservations may be written as:

`Name | Car Number | Reservation Period | Approval Time`

Collect both first name and surname for `Name`; exact rendering and output-file
layout remain to be chosen with the recording implementation. Pending/refused
requests are not approved records. Approval is not proof that recording succeeded.
The recording service must meet the assignment's security and reliability requirements.

SQLite is the authoritative operational reservation state. The Stage 3 file is an
approved-only output, not another source to synchronize/read for allocation. A failed
file write retains approval and allocation in SQLite and must be reported explicitly.
The dedicated disposable `.data/approved-reservations/` output directory will be
dropped/recreated at startup during Stage 3; that cleanup is not implemented yet.
Approval reversal is out of scope. Output-file layout and retries remain undecided.

Pending-request holds are selected for the later flow, before human review. Hold
timing, overlap eligibility, refusal handling, and on-demand validation/cleanup still
need agreement; no timers/background cleanup or reservation ledger is implemented here.
Communication channel and MCP implementation (or permitted fallback) remain open.

The user selected separate sequential one-space requests in one thread, not a single
multi-space request. The eventual SQLite ledger must account for that thread's earlier
blocking requests as well as other threads' requests. Atomic eligibility-check/hold
transactions, request-ID idempotency and half-open periods are recommendations for
later implementation; timing/overlap/refusal/release rules are not yet agreed.
