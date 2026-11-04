# Stage 4: system validation and load report

**Outcome:** the selected four-stage workflow is integrated and validated locally.
The original measured run passed 11 end-to-end scenarios, 90 concurrent-profile conversations,
a ten-way allocation race, and two live Azure/DIAL pipeline smoke conversations.
This is assignment-scale evidence, **not production capacity, administrator
authentication, complete confidentiality, or live-provider load testing**.

Run timestamps use the application machine's UTC clock. The original run was
**2026-10-31 UTC** (2026-11-01 local time): local validation ran 21:27:03–21:37:48
UTC; live smoke ran 21:39:33–21:40:56 UTC.

## Conversation regression validation — 2026-11-02

This regression covers browse continuity, automatic option rechecking, space references,
and named or vague corrections. These checks use bounded English/model extraction;
they do not establish universal semantic accuracy.

This evidence uses isolated synthetic data and the public input/review paths:

| Check | Result |
| --- | --- |
| Edge regressions | 27 fast injected-extraction tests, including bad outputs, negation, corrections, stale confirmation, failed revalidation, price/overlap changes, option continuity and async/thread isolation |
| Full HTTP/stdio integration | 13/13 scenarios, including approval/refusal and duration revalidation |
| Conversation-regression live components | 7/7 passed separately in this slice, inspecting parsed output before application safeguards/fallback |
| Live full pipeline | Two approval/refusal conversations passed in each of two runs, with real embeddings/extraction/review and actual stdio recording |
| Small local load regression | 6/6 conversations at concurrency 5: 3 approvals/files, 3 refusals, 3 recording attempts, zero retries/errors/overlaps; observed node concurrency 5 |
| Refreshed contention | 10 contenders: one pending winner, nine expected conflicts; refusal released the hold; no files |
| Source checks | Ruff lint/format, ty and `git diff --check` passed |

The approval/refusal scenario covers browsing, an offered space reference, duration
and personal-detail collection, correction and vague disagreement, confirmation,
submission of the retained eligible space, and review. It does not issue `/options`
or inject private state. The offline extractor is labeled; live runs use the real
extractor. Live component checks include negation and space-versus-vehicle extraction.
This seven-check slice predates the relevance-routing validation on the same date;
the latest documented offline and nine-check live-component results are in
[RAG_EVALUATION.md](RAG_EVALUATION.md#relevance-routing-validation-2026-11-02).

The integration run was **15:47:10–15:49:32 UTC**; before restart it had six approved
rows/files, zero overlaps and zero recorder children. Live repeats ran
**15:51:28–15:52:59** and **15:59:12–16:00:42 UTC** using `function_calling`,
`gpt-6-luna-2026-09-22`, and `text-embedding-3-small-1`. Each produced exactly one
approved file in one attempt; refusal produced none. Neither used fallback review wording;
both had zero node errors/overlaps and zero recorder children at inspection. Final live
dialogue median/p95 was 3.003/9.449 s (n=18); submission-to-review 2.924/3.467 s (n=2);
resume-to-outcome 1.603/2.258 s (n=2). These are tiny-sample observations, not capacity.

The small local load plus contention run was **15:59:12–16:00:53 UTC**. Its six
conversations took 22.518 s excluding 14.006 s startup (0.266 conversations/s); dialogue
median/p95 was 1.593/4.879 s (n=26). It is a regression probe on a shared host, **not a
rerun/replacement of the historical 90-conversation benchmark below**. Review remains
scripted; no fresh browser demonstration, human throughput, live-provider load,
resource-leak certification or complete privacy/security guarantee is claimed.

## Reproduce

Install the existing lockfile with Python 3.12+ and the local NLP model:

```bash
uv sync --locked --dev
# Focused harness contracts and real localhost HTTP/stdio integration
.venv/bin/python -m unittest discover -s tests -p 'test_system_validation.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_storage_imports.py' -v
# Full local system validation and 30 conversations at each concurrency level
PARKING_LIVE_TEST=0 LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
  .venv/bin/python -m system_validation --part offline --sessions 30 --concurrency 1 5 10
# Opt-in: existing .env credentials and VPN required; real models/embeddings/MCP
LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false PARKING_LIVE_TEST=1 \
  uv run --no-sync --env-file .env python -m system_validation --part live
```

The CLI prints JSON with environment, scenario outcomes, latency samples/counts,
throughput, recording attempts/retries, storage checks, and node concurrency. It
returns nonzero on a failed check, retains completed-part evidence in failure
reports, and emits only safe error/receipt codes—not private request IDs, raw
provider/protocol errors, chats, review payloads, or file contents.
`--part integration` and `--part load` select the corresponding local slices;
`--part all` includes the opt-in live slice. Sessions are bounded to 1–30, and CLI
load levels to 1/5/10. Smaller runs are probes, not substitutes for the reported run.

## Environment and method

| Item | Measured setup |
| --- | --- |
| Platform | Linux/local POSIX filesystem; Python 3.12.3; 8 logical CPUs |
| Frameworks | LangGraph 1.2.18, Agent Server 0.16.9, CLI 0.4.40 |
| Storage/recording | Chroma 1.5.9, official MCP SDK 2.3.0 |
| Provider adapter | langchain-openai 1.7.3 |
| Live model/embeddings | `gpt-6-luna-2026-09-22` / `text-embedding-3-small-1` |
| Structured extraction | `function_calling` |
| Load topology | One Agent Server process; 10 explicitly configured worker jobs |
| Test client | Real localhost TCP HTTP; stateful `/threads/.../runs/wait` and typed interrupt resumes |
| Recording | Actual short-lived stdio sessions and file publication, not a fake writer or HTTP MCP service |

`system_validation.py` starts the original `app:agent`/`app:webapp` application
with test-only model injection and node timing wrappers. It adds no public routes
or application interface. Server startup must finish before requests; a private
PID/port readiness marker correlates the owned server before health checking.

Every run owns a fresh `parking-stage4-*` temporary workspace for Chroma, SQLite,
approved output, and Agent Server checkpoints. The child verifies those storage
paths before loading the application. No project `.data/` path is used by the
harness. Local runs omit provider/tracing credentials and guard Python socket
connections against non-loopback destinations. Live runs forward only selected
provider settings; tracing remains disabled. Generated private data and diagnostics
are disposed at exit, including failure paths. These are test isolation measures,
not an OS sandbox.

Offline models are stateless: labeled, verbatim extraction; evidence-derived answer
text; neutral review wording; deterministic 64-dimensional hashed word embeddings.
Each synthetic model task includes 20 ms of asynchronous delay. This exercises
concurrent orchestration and real local privacy/storage work, **not real-model
quality, embeddings performance, or provider capacity**. Live smoke uses the real
configured models/embeddings, not those stubs or the 20 ms delay.

In the historical benchmark, each load conversation performed information dialogue,
collection, code confirmation, options, explicit submission, typed review, and
approval/recording or refusal. Turns within a thread were sequential; different
threads ran concurrently. Test choices came only from offered IDs. More-option turns
found distinct choices for each batch; they did not create holds or bypass the real
atomic submission check. Each profile had 15 approvals and 15 refusals, keeping the
20-space fixture within capacity while retaining prior approvals. Deliberate same-space
contention was measured separately, not counted as an unexpected system failure.

## Historical end-to-end scenarios: 11/11 passed (2026-10-31 UTC)

| Scenario | Checked outcome |
| --- | --- |
| RAG → multi-turn collection/correction → invalid review → approval | Missing facts unknown; private history filtered; current confirmation; fresh invalid-response interrupt; canonical pending hold; actual MCP/file and final success reply |
| Refusal | Hold released; no recording receipt/file |
| Public facts at the time | Python CRUD hours/prices appeared in current-turn evidence/answer; this predates the priced Chroma fixture |
| Duration changes | Old offers cannot submit after hours change and fresh confirmation |
| Permanent failure before writing | One attempt, explicit failure, approval retained, manual retry succeeds |
| Transient transport exhaustion | Two attempts, unconfirmed outcome, no ordinary-chat retry; manual retry retains original approval/period |
| Lost receipts after actual publication | One complete file despite replay; uncertainty is explicit; manual retry confirms matching content |
| Sequential `/new` | New request/receipt identity; earlier approved allocation unchanged |
| Public availability change | Submission rechecks blocking snapshot; no hold; restored default permits suggestions |
| Review after end | `elapsed`, not approved; no file |
| Server restart with old pending/approved checkpoints | SQLite/output reset; old checkpoints cannot resurrect requests/files; persistent 20-record Chroma collection reused |

Before the restart scenario: five approved rows, five complete files, zero
overlapping pending/approved allocations, zero recorder children. The assertion
that files match SQLite is **test verification**, not a new operational synchronization
or allocation source. Existing Stage 2/3 tests separately retain exact time-boundary,
replay, unsafe-link/path, unauthorized-call, timeout/cancellation, and streaming checks.

Fault cases explicitly inject client-side failures. The lost-receipt case performs
real stdio publication/replay before dropping receipts. Those fault latencies are
not mixed into the normal-load table.

## Historical normal load: 90/90 conversations passed (2026-10-31 UTC)

This benchmark predates the priced Chroma fixture and is not a priced-workflow load
run. The reproduction commands exercise the code present when they are run, not the
historical version automatically.

| Concurrent conversations | Completed | Approved/refused | Elapsed, excluding startup/warmup | Conversations/s | Observed simultaneous graph nodes |
| --- | --- | --- | --- | --- | --- |
| 1 | 30/30 | 15/15 | 220.815 s | 0.136 | 1 |
| 5 | 30/30 | 15/15 | 84.362 s | 0.356 | 5 |
| 10 | 30/30 | 15/15 | 77.783 s | 0.386 | 10 |

Each profile produced **15 exact approved files**, with 15 recording attempts,
**zero automatic recording retries, zero unexpected errors, zero overlapping
allocations, and zero unreaped recorder children** at end-of-run inspection.
Startup took 15.432/15.924/16.126 s respectively and was excluded from throughput.
Closed-loop timings include HTTP/queueing, privacy, test assertions, and coordination;
they are not isolated service-capacity estimates.

| Measurement | 1 concurrent: median / p95 | 5 concurrent: median / p95 | 10 concurrent: median / p95 |
| --- | --- | --- | --- |
| Dialogue turn | 0.916 / 1.242 s (n=120) | 1.328 / 2.110 s (n=132) | 1.993 / 4.087 s (n=156) |
| Submit → review interrupt | 0.786 / 1.141 s (n=30) | 1.261 / 1.797 s (n=30) | 1.900 / 2.332 s (n=30) |
| Resume → final outcome | 1.970 / 3.409 s (n=30) | 1.949 / 4.083 s (n=30) | 2.505 / 5.313 s (n=30) |
| Whole conversation | 7.417 / 8.705 s (n=30) | 12.742 / 14.560 s (n=30) | 23.606 / 26.369 s (n=30) |
| Administrator wording node | 0.054 / 0.071 s (n=30) | 0.255 / 0.336 s (n=30) | 0.439 / 0.799 s (n=30) |
| Review/decision node, including paused/resumed executions | 0.011 / 0.084 s (n=60) | 0.033 / 0.214 s (n=60) | 0.034 / 0.221 s (n=60) |
| Actual stdio recording node | 2.213 / 2.721 s (n=15) | 2.840 / 3.299 s (n=15) | 3.244 / 4.306 s (n=15) |

Scripted review delay is measured separately: about 50 ms per conversation. It is
not human decision time or administrator throughput. Percentiles use nearest rank;
small samples, especially n=15 recording measurements, make p95 close to the maximum.
Raw stdout JSON includes maxima, node distributions, and exact sample counts.

The ten-way same-space race had **one pending winner and nine correctly reported
conflicts**. No file was written while pending; refusal released the winner's hold.

### Load limits

This is a shared development host with other processes, not an isolated benchmark
machine. Throughput barely improved from five to ten clients while conversation
latency rose. Local privacy detection serializes NLP work, and stdio includes cold
SDK/process startup; the data does not establish either as the sole bottleneck.
File-descriptor snapshots were 19→20, 19→27, and 19→22. They are observations during
API workloads, **not proof of no descriptor/memory leaks** or a sustained soak/peak
resource measurement. Owned server processes and diagnostic handles are closed;
recorder children were absent at inspection. Do not infer a production SLA.

## Historical live pipeline smoke: 2/2 passed (2026-10-31 UTC)

Real temporary Chroma seeding/retrieval, the then-current extraction and administrator
models, typed review, SQLite and actual stdio MCP were used, with tracing disabled:

- **Approval:** RAG questions, duration-only collection, surname correction,
  stale-confirmation rejection, disjoint other options, empty review response/retry,
  human-decision-path approval, one recording attempt, exact file and success reply.
- **Refusal:** full collection/submission/review; refusal, no file, hold released.
- Both reviews used accepted generated wording rather than the fixed fallback;
  canonical facts matched SQLite. Zero graph-node errors, zero overlapping
  allocations, one approved row/file, zero remaining recorder children.

Live dialogue median/p95 was 2.614/7.964 s (n=13); whole conversation was
31.099/42.953 s (n=2). Administrator composition was 2.147/2.180 s (n=2), and the
single actual recording node took 4.078 s. These tiny-sample timings are **smoke
observations only**. No live-provider concurrent load or fresh browser demonstration
was performed; review responses were scripted through the Studio-compatible interrupt.

## Requirement coverage and delivery

| Stage 4 baseline requirement | Implementation/evidence |
| --- | --- |
| LangGraph orchestration and complete workflow | Graph in `parking_agent.py`; lifecycle/hold/recording logic documented in README |
| Entire workflow integration testing | Dated real HTTP/server/stdio scenarios plus sync/async graph and actual server tests |
| Dialogue, administrator confirmation and recording load | Historical three-profile benchmark, node distributions, real typed resumes and stdio, separate contention race; later small regression |
| Usage, architecture, agent/server logic, setup and deployment | README, PARKING_DATA_MODEL.md, this report; selected Linux/local Studio deployment only |
| Repository link for each stage | README submission table; publication/answer-field submission remains an operator action |

RAG quality and privacy limitations are documented in RAG_EVALUATION.md and
GUARDRAIL_EVALUATION.md.
