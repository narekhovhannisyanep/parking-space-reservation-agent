# Privacy guardrail evaluation

This is a small synthetic, local privacy evaluation of the implemented guardrails,
not the assignment's complete RAG accuracy/performance report, a load test, or a
security certification. No Azure/DIAL calls or real customer data are used.

## Reproduce

```bash
uv sync --dev
.venv/bin/python -m guardrail_evaluation
.venv/bin/python -m unittest discover -s tests -p 'test_guardrail*.py' -v
```

The locked model is `en_core_web_sm` 3.8.0; tested with spaCy 3.8.18 and
presidio-analyzer 2.2.366. Setup installs the model explicitly. Detection cannot
download models or domain lists at runtime. The small CPU model is an assignment-
appropriate starting point; it is not a comprehensive multilingual PII model.

## Method

`guardrail_evaluation.py` contains 21 reviewable synthetic cases: 13 with protected
values and eight benign parking messages. They cover English names, email, phone,
card/IBAN, credential patterns, mixed PII, known vehicle identifiers, an Armenian
name, and public hours/prices/directions/periods. Known and unknown identifiers are
tested separately. The corpus deliberately includes detection limitations.

A positive case passes only if **every annotated character of every protected
value, including repeated occurrences, is covered by detection**. Partial detection
counts as a miss. A benign case with any span counts as a false positive. Reported
precision and recall are these case-level coverage metrics, not token/entity-level
NER scores. Case IDs/counts are printed, not the input text or detected values.

Latency measures one warm detector call per case after model preparation; it excludes
model loading, storage, embeddings, LLM calls, queuing, and complete chat latency.
This is not an independent held-out benchmark; the corpus includes regression
examples used while tuning narrow public-data false-positive rules.

## Observed results (run date not recorded)

| Metric | Result |
| --- | --- |
| Fully protected positive cases | 12 / 13 |
| Missed/partially protected positive cases | 1 / 13 |
| Benign cases falsely flagged | 0 / 8 |
| Benign cases accepted | 8 / 8 |
| Case-level precision | 100% |
| Case-level recall | 92.31% |
| Warm mean detector latency | 12.704 ms (local run; varies by hardware/workload) |

`unknown-vehicle` was missed: an unlabeled `blue bicycle 7` cannot reliably be
identified as private by a general English NER model. The corresponding known-draft
case was fully protected. The one Armenian example was detected in this run; that
is **not** evidence of dependable Armenian or multilingual coverage.

All 20 seed documents and benign public hours/prices passed real local guard checks.
Tests separately verify deterministic checks for known Unicode/nonplate identifiers,
overlapping spans, nested JSON keys and metadata, whole-batch rejection before
embedding, contaminated existing evidence, extraction grounding after redaction,
old-draft privacy after cancellation, detector/missing-model failure, and sync/async
`messages` streaming.

## Boundaries and limits

- Raw necessary reservation details still reach the configured extraction provider.
  Code-generated summaries show the current user's validated details.
  Extraction permits PERSON spans to interpret names; it does not reliably classify
  whether every name in the latest message is necessary. Unrelated names may reach
  extraction too. Detected non-name PII/credential patterns are filtered there.
- General-information model inputs and generated responses are sanitized. Successful
  draft names/vehicle identifiers are retained thread-locally for filtering later
  turns even after corrections, cancellation, or a new draft. This is not deletion.
- Public ingestion is checked before embeddings/writes; selected existing evidence is
  checked before graph context/model use. Reused Chroma collections are not scanned
  in full, migrated, or modified automatically.
- General NER can miss names, arbitrary identifiers, obfuscation, and secrets without
  recognized patterns. It can also falsely flag legitimate public prose. Source
  review and the public/private storage split remain essential.
- Exact generated inventory prefixes and fixed schema key labels have narrow
  structural handling; metadata values/arbitrary keys remain checked. Numeric
  technical IDs and explicit numeric currency phrases avoid known PERSON false
  positives. These exceptions do not make arbitrary source text trustworthy.
- `nostream` prevents raw model tokens in LangGraph's `messages` stream; final graph
  responses are inspected. It does **not** protect raw user inputs, checkpoints,
  Studio Graph/debug/task/state APIs, `astream_events`, or enabled tracing. These
  remain trusted development surfaces, not a multi-user authorization boundary.
- Keep tracing disabled and the server local. This slice adds neither authentication,
  encryption/retention controls, a prompt-injection classifier, nor complete output
  grounding. Prompts are defense-in-depth, not proof of confidentiality or approval.

References: [Presidio evaluation](https://data-privacy-stack.github.io/presidio/evaluation/),
[LangChain guardrails](https://docs.langchain.com/oss/python/langchain/guardrails),
[LangGraph streaming](https://docs.langchain.com/oss/python/langgraph/streaming).
