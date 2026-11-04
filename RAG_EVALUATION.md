# RAG evaluation

This is a small synthetic evaluation of the Stage 1 RAG chatbot. It covers retrieval
quality, request detection, "other options" follow-ups, rule-checked answers, and
latency for each node. It runs the real LangGraph pipeline against temporary
Chroma/SQLite copies of the bundled 20-space mock inventory. It is not a
statistically powered benchmark, a load test, or proof of confidentiality. Privacy
detection is evaluated separately in [GUARDRAIL_EVALUATION.md](GUARDRAIL_EVALUATION.md).

## Reproduce

```bash
# Offline harness/metric tests (no provider calls)
.venv/bin/python -m unittest discover -s tests -p 'test_rag_evaluation.py' -v
# Live run: needs the VPN, .env credentials and LANGSMITH_TRACING=false
uv run --no-sync --env-file .env python -m rag_evaluation --part retrieval   # embeddings only
uv run --no-sync --env-file .env python -m rag_evaluation --part all         # + chat/extraction models
```

The CLI prints a JSON report and uses temporary Chroma/SQLite stores. It does not
write a persistent report or touch the project `.data/` stores.

## Method

| Part | What it measures | Models |
| --- | --- | --- |
| Retrieval | Recall@3, capped Recall@3, Precision@3, Hit@3 and MRR over the original 20 hand-labeled queries | Real embeddings; stub chat/extraction |
| Practical benefits | 8 additional hand-labeled route/usefulness requests, including 2 challenge paraphrases | Same run; reported separately |
| Budgets | 12 explicit upper-limit/range/clarification cases: compliance, count, price display, blocked inventory | Same run; Python parses cues independently of stub extraction |
| Request detection | Whether 14 messages correctly do, or do not, produce availability options | Same run |
| Follow-ups | Whether 3 "other options" conversations offer new, nonoverlapping spaces | Same run |
| Answers | 13 conversations checked with deterministic rules | Real embeddings, chat and extraction |
| Latency | Wall-clock time per graph node and per turn (median, nearest-rank p95, mean) | Both runs |

- **Labels.** Each retrieval query is labeled with every fixture space whose public
  description matches the stated preference. For example, "near the elevator" is labeled
  `G-E02, G-E03, L1-E01, L2-E01`. Several queries have more than three relevant
  spaces, so plain Recall@3 cannot reach 1.0 for them. **Capped recall** divides the
  hits by `min(3, relevant)` and is the fairer "best possible" view. Precision@3 is
  capped by the label size in the same way: a query with one relevant space scores
  at most 0.333.
- **Answer rules.** Answers are checked by rules rather than an LLM grader, so the
  results are reproducible, explainable and add no cost:
  - SQLite hours and Chroma hourly-rate metadata summaries are quoted;
  - missing hours are stated as unknown;
  - offered IDs are mentioned;
  - no unoffered inventory IDs or labels appear;
  - no reservation, booking or approval is claimed;
  - known private names are not echoed;
  - follow-up options don't overlap earlier ones.

  Rules can misjudge paraphrases, so failures were inspected by hand (see below).
- **Facts.** Hours/availability are written through Python CRUD into temporary SQLite
  and removed afterwards. Hourly prices come only from Chroma metadata. One case
  injects contradictory legacy SQLite prices and verifies they are ignored; empty-
  inventory price handling has separate unit coverage.
- **Latency.** Latency includes the provider network round trip, which this evaluation
  does not control. Retrieval-only turns use stub models, so `parking_assitant` and
  `reservation_collection` reflect local overhead only.

## Priced-inventory results (2026-11-02)

The measured fixture contains 20 differentiated location/route descriptions,
practical benefits and trade-offs. Required fixed USD rates are numeric Chroma
metadata only (`hourly_price_cents=100..2000`, schema 3),
not embedded prices. Queries embed code-owned public terms, including
mobility, shopping bags, pushchairs, weather, traffic and departures. Negative stairs
requests map to step-free access. All evaluation stores were temporary.

The live run used `text-embedding-3-small-1` and `gpt-6-luna-2026-09-22`, tracing off,
with `function_calling`. Two answer runs both passed 12/13 cases. The following is
one retrieval measurement, not a statistically powered comparison:

| Original 20-query set, K=3 | Result |
| --- | --- |
| Recall@3 | 0.547 |
| Capped Recall@3 | 0.625 |
| Precision@3 | 0.500 |
| Hit@3 | 0.750 |
| MRR | 0.567 |
| Request detection (14) | 14/14 |
| Disjoint other-options conversations (3) | 3/3 |

### Practical benefits: 8 separate queries

Aggregate capped Recall@3 / Precision@3: **0.583**; MRR: **0.604**.

| Case | Offered IDs | Capped recall |
| --- | --- | --- |
| Wheelchair route | G-E02, G-E03, L1-B02 | 0.667 |
| Avoid stairs | G-E02, G-E03, L1-E01 | 1.000 |
| Shopping bags | OUT-V01, G-A02, OUT-P01 | 0.667 |
| Stroller | OUT-P01, G-A02, L1-E01 | 1.000 |
| Sheltered walk in rain | OUT-C01, OUT-P01, G-A02 | 0.333 |
| Away from traffic | OUT-C01, OUT-C02, G-N01 | 1.000 |
| “Trouble walking” challenge | OUT-V01, G-A01, G-A04 | 0.000 |
| “Won't get soaked” challenge | G-A04, OUT-V01, OUT-V02 | 0.000 |

The two challenge paraphrases were frozen before measurement. They are a tiny
out-of-vocabulary probe, not an independent statistical hold-out benchmark.
Rain/unloading negatives can rank highly because they mention the same concepts as
positive requests. Descriptions remain semantic preferences, not hard accessibility/
weather constraints or a certification of route suitability.

### Budget compliance and live extraction

All **12/12 budget cases passed**: strict/inclusive $8 bounds, inclusive $5–$10 range,
$8.50 and word-valued limits, a $0.50 cap with no matches, all-blocked inventory, and
clarification for total-stay, omitted-unit, non-USD, reversed-range and fuzzy budgets.
Each case checks allowable IDs, expected up-to-three result count, per-option USD/hour
display and clarification state; empty results are checked separately, not counted as
evidence of successful nonempty retrieval. Budget filters and eligible IDs apply before
Chroma ranking and its matching-record count. Rates never come from SQLite snapshots.

Retrieval evaluation uses stub extraction, so this proves deterministic price enforcement
for the tested grammar, **not real-model extraction accuracy**. A separate opt-in live
component test verified that the model returned complete grounded budget phrases for
strict/inclusive upper limits and a range; Python independently interpreted them. All
five live component checks passed (existing extraction/collection/administrator checks
plus budget extraction). This is smoke coverage, not universal free-form understanding.

### Live answers and latency

Both runs: **12/13 conversations fully passed**. All four offer-list checks mentioned
every offered option; all five unoffered-ID checks passed; all twelve no-booking-claim
checks passed. Hours/prices and legacy-price precedence passed. The remaining failure
is the known exact-ID question “Where is space E02?”: fixed topic queries do not perform
an exact-ID lookup.

Application code appends every validated offered ID/description/rate to guarded
model replies as well as collection/control replies. That completeness improvement is
**code enforcement, not evidence that the answer model learned to list all options**.
General model prose remains prompt-grounded; this is not complete output-grounding or
prompt-injection protection.

Latest of the two measured answer runs (14 turns): extraction median 1.788 s / p95
2.969 s; context median 0.714 s / p95 1.697 s; response median 2.536 s / p95 11.944 s;
whole turn median 5.574 s / p95 14.420 s. Retrieval-only context (56 executions) median
0.616 s / p95 1.500 s. Provider latency varies and the shared-host/sample size is small;
these are not load-capacity measurements.

Offline tests additionally cover budget persistence/clearing, confirmation independence,
unsupported/missed/forged extraction, boundary comparisons, availability intersection,
submission rate rechecks, async/thread isolation, and same-ID cross-schema upsert refusal.
A real localhost HTTP Agent Server test carries a budget-filtered option through typed
approval and actual stdio MCP recording with exact output checks. The historical
90-conversation Stage 4 benchmark is not a priced-workflow load run.

## Relevance-routing validation (2026-11-02)

The structured extraction schema distinguishes parking information/reservation from
`unrelated` turns. Python skips retrieval and clears per-turn evidence for unrelated
messages, preserving private drafts, confirmation, browse options and optional budgets.

Offline regressions cover sync/async bypass, shared words (restaurant hours/prices/book),
stale-evidence clearing, budget/confirmation/option continuity and bounded guarded
history. The complete suite passed **391 tests, with nine opt-in live tests skipped**
(400 total, 312.276 s). Ruff lint/format and ty checks passed.

All **nine live component checks passed** separately (65.799 s), using the existing
`function_calling` / `gpt-6-luna-2026-09-22` configuration with tracing disabled and
mocked stores. The relevance test checked eight explicit cases: three unrelated
restaurant requests, a restaurant-visit parking request, general parking information,
opening-hours follow-ups after parking and restaurant discussion, and other parking
options. A restaurant request plus its hours follow-up also passed through the real
model graph with no storage retrieval, empty evidence and parking-scope replies.

These are small synthetic smoke checks, **not universal intent accuracy**, new browser
validation or live-provider load evidence. Relevance and answer wording remain
model-guided; a misclassification can still select the wrong branch. Retrieval quality
and latency tables above predate this routing correction and were not remeasured.

## Earlier Stage 1 results (2026-10-30)

On the earlier 20-space short-description fixture, using `gpt-6.1-sol-2026-09-29`
and `text-embedding-3-small-1`, the 20-query retrieval run measured Recall@3 0.535,
capped Recall@3 0.608, Precision@3 0.483, Hit@3 0.800 and MRR 0.692. Request
detection passed 14/14 messages and other-options follow-ups 3/3 conversations.
Elevator synonyms and some ground-level or one-word location queries remained weak.
These are historical results on a different fixture, not current performance.

The rule-checked live answer run passed 10/13 conversations: two failed to mention
all offered options and one failed the exact-ID location question. It used the then
current hours/prices behavior, including missing-price checks. In the live answer run
(node/turn samples n=13–14), median/p95 latency was 2.54/5.51 s for extraction,
0.51/1.42 s for context, 2.83/4.28 s for response, and 5.38/9.79 s per turn.
Retrieval-only context latency was 0.46/0.61 s (n=41). These small, provider-dependent
samples are not capacity measurements; the dated Stage 4 results use a different
full-workflow method.

## Limits

- The corpus is small, synthetic and English-only. Labels are developer-written, and
  the queries include examples used in developing the vocabulary; results are not an
  independent estimate of generalization.
- Live results are non-deterministic and depend on the provider and its version.
  Repeated runs can vary.
- Rule-based answer checks can misjudge paraphrases. They don't measure helpfulness,
  tone or completeness.
- Stage 1 retrieval and answer measurements do not measure period allocation or
  approved recording and are not load tests. Dated full-system local concurrency and
  live pipeline smoke results are reported separately in STAGE4_VALIDATION.md.
