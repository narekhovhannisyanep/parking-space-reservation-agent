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

The CLI prints a JSON report. It never touches the project `.data/` stores and writes
no files.

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
  injects contradictory legacy SQLite prices and verifies they are ignored. The
  earlier missing-price case is now this legacy-precedence case because prices are
  required in the fixture; empty-inventory price handling has separate unit coverage.
- **Latency.** Latency includes the provider network round trip, which this evaluation
  does not control. Retrieval-only turns use stub models, so `parking_assitant` and
  `reservation_collection` reflect local overhead only.

## Current priced-inventory results (2026-11-02)

The fixture now contains 20 realistic, differentiated location/route descriptions,
practical benefits and trade-offs, preserving every space ID/location. Required fixed
USD rates are numeric Chroma metadata only (`hourly_price_cents=100..2000`, schema 3),
not embedded prices. Queries still embed code-owned public terms, now also covering
mobility, shopping bags, pushchairs, weather, traffic and departures. Negative stairs
requests map to step-free access. No new package, provider, real `.env` edit or populated
project-data reseeding/deletion was needed. All evaluation stores were temporary.

The live run used `text-embedding-3-small-1` and `gpt-6-luna-2026-09-22`, tracing off,
with `function_calling`. Two answer runs both passed 12/13 cases. The following is
one retrieval measurement, not a statistically powered comparison:

| Original 20-query set, K=3 | Earlier short fixture (2026-10-30) | Detailed priced fixture |
| --- | --- | --- |
| Recall@3 | 0.535 | 0.547 |
| Capped Recall@3 | 0.608 | 0.625 |
| Precision@3 | 0.483 | 0.500 |
| Hit@3 | 0.800 | 0.750 |
| MRR | 0.692 | 0.567 |
| Request detection (14) | 14/14 | 14/14 |
| Disjoint other-options conversations (3) | 3/3 | 3/3 |

**Results are mixed, not a uniform retrieval improvement.** Recall/precision increased
slightly; hit rate and first-relevant ranking decreased. Longer realistic descriptions
provide more answerable context, but mentions of route limitations can compete with
positive benefits in embeddings. The old labels/query set are retained for visibility;
the richer corpus expands some meanings, so the historical comparison is imperfect.
We did not rewrite the fixture/query vocabulary to erase challenge failures.

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

The two challenge paraphrases were frozen before measurement and were not added to
the mapper after failing. They are a tiny out-of-vocabulary probe, not an independent
statistical hold-out benchmark. Rain/unloading negatives can rank highly because they
mention the same concepts as positive requests. Descriptions remain semantic preferences,
not hard accessibility/weather constraints or a certification of route suitability.

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
an exact-ID lookup. No exact-ID feature was added to make this test pass.

Application code now appends every validated offered ID/description/rate to guarded
model replies as well as collection/control replies. That completeness improvement is
**code enforcement, not evidence that the answer model learned to list all options**.
General model prose remains prompt-grounded; this is not complete output-grounding or
prompt-injection protection.

Latest of the two measured answer runs (14 turns): extraction median 1.788 s / p95
2.969 s; context median 0.714 s / p95 1.697 s; response median 2.536 s / p95 11.944 s;
whole turn median 5.574 s / p95 14.420 s. Retrieval-only context (56 nodes) median
0.616 s / p95 1.500 s. Provider latency varies and the shared-host/sample size is small;
these are not load-capacity measurements.

Offline tests additionally cover budget persistence/clearing, confirmation independence,
unsupported/missed/forged extraction, boundary comparisons, availability intersection,
submission rate rechecks, async/thread isolation, and same-ID cross-schema upsert refusal.
A real localhost HTTP Agent Server test carries a budget-filtered option through typed
approval and actual stdio MCP recording with exact output checks. Existing Stage 4
load evidence remains historical and is not claimed to be a new priced-workflow load run.
The final offline regression completed **359 tests: 354 passed, five opt-in live
component checks skipped**. Ruff lint/format and ty checks passed. Long non-cent
decimal amounts are rejected by inspecting their original digits, not rounded with
a floating-point or decimal precision context.

## Historical results (2026-10-30, `gpt-6.1-sol-2026-09-29`, `text-embedding-3-small-1`)

### Retrieval, before and after the query-vocabulary change

The baseline was measured with the earlier code: six fixed topic terms, and no
"spot"/"park"/"vacant" request words.

| Metric (K=3, 20 queries) | Before | After |
| --- | --- | --- |
| Recall@3 | 0.160 | **0.535** |
| Capped Recall@3 | 0.183 | **0.608** |
| Precision@3 | 0.150 | **0.483** |
| Hit@3 (at least one relevant) | 0.35 | **0.80** |
| MRR | 0.317 | **0.692** |
| Request detection accuracy (14) | 0.714 (recall 0.50) | **1.000** |
| "Other options" follow-ups (3) | 0 / 3 | **3 / 3** |

Before the change, 11 of 20 queries embedded the identical string `availability` and
received the same three spaces. Three "spot" requests received no options at all.
After the change, location preferences become code-owned terms such as `stairs`,
`outdoor area`, `second level` or `north wall`. Stairs, outdoor, second-level, wall,
lobby and fence queries now reach perfect capped recall.

Remaining weak cases after the change:

| Case | Query embedded | Offered | Observation |
| --- | --- | --- | --- |
| elevator / lift / need-spot-elevator | `lift availability` | G-E02, G-A04, G-A01 | Only one fixture text uses "lift"; the others say "elevator", so synonyms rank lower |
| ground-elevator-free | `lift ground level availability` | L2-B01, L1-E01, L1-B02 | "level" dominates; few ground-floor descriptions say "ground" |
| corner, walkway, west-entry | e.g. `corner availability` | unrelated | One-word topics are weak signals for short descriptions |
| east-side | `east availability` | 1/3 relevant | Same weakness |

Possible next tuning steps, not implemented, to avoid fitting this small labeled set:
- emit both `lift elevator`;
- drop the generic `availability` word when a location term exists;
- enrich fixture descriptions.

Any tuning should be re-measured with a held-out query set.

### Answers (13 live conversations)

| Check | Pass |
| --- | --- |
| No booking/approval claim (incl. "please book now" and prompt-injection attempts) | 12 / 12 |
| No unoffered inventory IDs/labels | 5 / 5 |
| Offers non-empty when eligible / empty when all blocked | 4 / 4, 1 / 1 |
| Follow-up options disjoint | 1 / 1 |
| Missing hours/prices stated as unknown | 2 / 2 |
| Known private name not echoed | 1 / 1 |
| Stored hours/prices quoted | 2 / 2 |
| All offered options mentioned | 2 / 4 |
| Exact-space location lookup ("Where is space E02?") | 0 / 1 |
| **Cases fully passed** | **10 / 13** |

Failures, inspected by hand:

- **Not all offered options are mentioned (2 cases).** The model named only the best
  elevator match (E02 / E01-L1) and left out the other two code-selected options,
  which aren't near the elevator. That's arguably helpful, but users don't see every
  offer in the model's text. Deterministic collection replies already list all
  options; information replies rely on the prompt.
- **Exact-ID lookup (1 case).** "Where is space E02?" embeds only `location directions`,
  so E02 isn't retrieved, and the model correctly says it has no details instead of
  inventing directions. Exact-ID search was deliberately not implemented, and fixed
  queries never embed user-supplied IDs. This case documents that limit.
- A first live run reported 0/2 for "unknown" statements. Inspection showed correct
  answers written with typographic apostrophes ("don’t", "aren’t") that the rule
  missed. The rule was fixed, covered by a unit test and the run repeated. Results
  above are from that repeated run.

### Latency

| Node / turn (live answer run, n=13–14) | Median | p95 |
| --- | --- | --- |
| `reservation_collection` (structured extraction model call) | 2.54 s | 5.51 s |
| `context_augmentation` (SQLite read + query embedding + Chroma) | 0.51 s | 1.42 s |
| `parking_assitant` (answer model call; ~0 for deterministic replies) | 2.83 s | 4.28 s |
| Whole turn | 5.38 s | 9.79 s |

In the retrieval-only run (n=41), `context_augmentation` had a median of 0.46 s and
a p95 of 0.61 s. Nearly all of that is the embedding round trip; local
Chroma/SQLite work is a few milliseconds. Seeding the temporary collection with 20
documents took about 3–5 s.

Information turns in this historical Stage 1 run made two sequential model calls:
extraction, then answer. Current deterministic collection/control/outcome turns do
not always call both models. An earlier answer run on the same day saw a 63 s p95
from one slow provider response, so provider latency varies. Current full-workflow
timings are reported separately in STAGE4_VALIDATION.md.

## Limits

- The corpus is small, synthetic and English-only. The labels were written by the
  developer from 20 short descriptions. The before/after comparison used the same
  queries the fixes were designed for, so the absolute "after" numbers are
  optimistic. The baseline weaknesses, though, are reproducible.
- Live results are non-deterministic and depend on the provider and its version.
  Repeated runs can vary.
- Rule-based answer checks can misjudge paraphrases. They don't measure helpfulness,
  tone or completeness.
- These measurements preceded the private hold ledger and duration-only workflow.
  They do not measure current period allocation or approved recording; the ledger
  is now implemented and validated separately in STAGE4_VALIDATION.md.
- This historical Stage 1 evaluation is not a load test. Current full-system local
  concurrency/throughput and live pipeline smoke are reported in STAGE4_VALIDATION.md;
  those results do not replace the retrieval-quality measurements above.
