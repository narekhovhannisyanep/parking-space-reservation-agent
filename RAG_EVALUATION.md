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
| Retrieval | Recall@3, capped Recall@3, Precision@3, Hit@3 and MRR over 20 hand-labeled preference queries | Real embeddings; stub chat/extraction |
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
  - stored hours/prices are quoted;
  - missing facts are stated as unknown;
  - offered IDs are mentioned;
  - no unoffered inventory IDs or labels appear;
  - no reservation, booking or approval is claimed;
  - known private names are not echoed;
  - follow-up options don't overlap earlier ones.

  Rules can misjudge paraphrases, so failures were inspected by hand (see below).
- **Facts.** Facts are written through the existing Python CRUD into the temporary
  SQLite store for each case and removed afterwards, matching the empty-at-startup
  design.
- **Latency.** Latency includes the provider network round trip, which this evaluation
  does not control. Retrieval-only turns use stub models, so `parking_assitant` and
  `reservation_collection` reflect local overhead only.

## Results (2026-10-30, `gpt-6.1-sol-2026-09-29`, `text-embedding-3-small-1`)

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
