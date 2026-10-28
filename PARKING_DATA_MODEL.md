# Parking data model

## Scope

The four stages in `AGENTS.md` are the goal: RAG/chat, human administrator approval,
approved-reservation recording, and LangGraph orchestration with testing/evaluation.
Do not add real-world parking management features unless they become requirements.

**Project assumption:** every space fits any vehicle, whether a bicycle, car, or semi.
There are no vehicle categories, dimensions, clearance checks, or facility requirements.
This is a simplifying prototype assumption, not a real-world safety guarantee.
Spaces retain individual IDs so code can allocate one without double booking.

## Static inventory (implemented)

`storage/parking_spaces.json` contains eight mock spaces at fictional `demo-central`.
Each record requires exactly these fields:

| Field | Purpose |
| --- | --- |
| `space_id` | Stable space identity within a location |
| `location_id` | Explicit location scope |
| `label` | Human-visible space marking |
| `description` | Short public directions/information for retrieval |
| `source` | Reviewed source attribution |

All values are trimmed, nonblank strings. IDs use 1–64 alphanumeric/dash/underscore
characters and start alphanumerically. Description is limited to 400 characters;
other text to 200. Unknown/private fields are rejected, but text still needs review
for sensitive content and source authenticity.

`ParkingSpace` generates a LangChain document with ID `{location_id}:space:{space_id}`.
Metadata contains the fields except description, plus `record_type=parking_space`
and integer `schema_version=2`. `SpaceFilters` supports only optional exact `space_id`;
every search filters location, record type, and schema before semantic ranking.
Directions need no separate level/zone fields. Vehicle identification is never indexed.

The default collection is `parking_spaces_v2`. Re-seed into this new collection if
an older collection was used. Do not upsert new records over old-schema IDs in the
old collection: Chroma merges metadata and could retain removed restrictions.
Existing `.env` and databases are unchanged; see the README migration instructions.

## Public operational facts (implemented storage only)

`DynamicParkingStore` stores location-scoped `working_hours`, `prices`, and
`availability` as JSON snapshots in SQLite, with UTC write timestamps. It remains
unseeded and is not an authoritative booking ledger. Missing data means unknown.
No customer details, administrator decisions, or reservation requests belong here
or in Chroma. Required general information and location facts must also be supplied
when wiring RAG; the space fixture alone does not cover every required question.

## Reservation data (future, minimal contract)

Keep private requests separate from public facts. A request needs:

- A code-generated request ID and workflow status.
- First name, surname, `car_number`, and reservation start/end.
- Location ID and the space ID assigned by code.
- Trusted administrator approval/refusal and approval time.
- Recording status/idempotency reference so retries do not duplicate file entries.

`car_number` remains the required field/file heading, but means the requester's
vehicle identifier. Accept a user-provided identifier for vehicles without plates;
do not require a vehicle type, measurements, or a plate format, and do not invent
an identifier. There is no separate vehicle entity or fit/feature-matching service.

Code must validate the required details and time period, prevent overlapping
bookings for the same space, and accept approval only through the trusted human
workflow. Resolve an explicit timezone when interpreting dates; normalize stored
instants to UTC. Half-open intervals `[start, end)` allow adjacent bookings.

Pending-request holds are still under consideration, not a required implemented
feature. Choose holds or an atomic availability recheck before confirmation when
implementing allocation. Never hold a database transaction open while awaiting
the human, or treat top-K retrieval as an exhaustive availability result.

Only approved reservations can reach the recording service. The required format is
`Name | Car Number | Reservation Period | Approval Time`, with Name composed from
first name and surname. Refused/pending requests are not approved bookings; approval
is not proof of successful recording. MCP or a documented permitted fallback,
authorization, privacy protection, and retry-safe writes remain required work.

## Lifecycle and status

Initialization, reviewed seeding, readiness validation, and refresh remain separate.
Imports/chat do not seed databases. Static knowledge-change approval is separate
from reservation approval; the seed CLI is trusted ingestion, not an approval system.

The graph currently does not retrieve storage facts, allocate, contact an
administrator, or record reservations. Detailed production inventory mirrors,
occupancy feeds, chargers, maintenance tables, tariffs, and hold-renewal services
are outside the current scope. Implement only what is needed for the four stages.

Reference: [Chroma metadata filters](https://docs.trychroma.com/docs/querying-collections/metadata-filtering)
and [LangChain Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma).
