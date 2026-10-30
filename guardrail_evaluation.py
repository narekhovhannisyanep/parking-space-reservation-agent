"""Small synthetic privacy evaluation, not live RAG accuracy or a security certification."""

import json
import re
from dataclasses import dataclass
from importlib.metadata import version
from statistics import mean
from time import perf_counter

from guardrails import MODEL_NAME, PrivacyGuard


@dataclass(frozen=True)
class Case:
    identifier: str
    text: str
    protected: tuple[str, ...] = ()
    known: tuple[str, ...] = ()


CASES = (
    Case("name-full", "My name is John Smith.", ("John Smith",)),
    Case("name-followup", "Alice Smith asks for parking.", ("Alice Smith",)),
    Case("email", "Email john@example.com", ("john@example.com",)),
    Case("phone", "Call +1 212-555-5555", ("+1 212-555-5555",)),
    Case("card", "My card is 4111 1111 1111 1111", ("4111 1111 1111 1111",)),
    Case("iban", "IBAN GB82 WEST 1234 5698 7654 32", ("GB82 WEST 1234 5698 7654 32",)),
    Case("secret-key", "api_key=synthetic-secret-value", ("synthetic-secret-value",)),
    Case("bearer", "Bearer synthetic-secret-token", ("synthetic-secret-token",)),
    Case("mixed", "John Smith email john@example.com near the entrance", ("John Smith", "john@example.com")),
    Case("known-vehicle", "blue bicycle 7 near entrance", ("blue bicycle 7",), ("blue bicycle 7",)),
    Case("known-unicode", "Անահիտ asks for parking", ("Անահիտ",), ("Անահիտ",)),
    # Deliberate coverage limitations, not cases to exclude to inflate recall.
    Case("unknown-vehicle", "blue bicycle 7 near entrance", ("blue bicycle 7",)),
    Case("unknown-unicode", "Անահիտ asks for parking", ("Անահիտ",)),
    Case("hours", "Open 09:00–18:00."),
    Case("prices", "Price EUR 9 per hour."),
    Case("directions", "The lift is beside the east entrance."),
    Case("location", "The parking location is demo-central."),
    Case("space", "Space A01 is near the entrance."),
    Case("period", "2030-06-01 10:00 to 2030-06-01 12:00 UTC+04:00"),
    Case("unknown-hours", "Working hours and prices are unknown."),
    Case("capacity", "Three parking options are suggestions, not holds."),
)


def evaluate(guard: PrivacyGuard, cases: tuple[Case, ...] = CASES) -> dict:
    guard.prepare()
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    failures: list[dict[str, str]] = []
    durations = []
    for case in cases:
        start = perf_counter()
        spans = guard.spans(case.text, known_values=case.known)
        durations.append((perf_counter() - start) * 1000)
        coverage = {index for span in spans for index in range(span.start, span.end)}
        expected = {
            index
            for value in case.protected
            for match in re.finditer(re.escape(value), case.text)
            for index in range(match.start(), match.end())
        }
        if case.protected:
            outcome = "tp" if expected and expected <= coverage else "fn"
        else:
            outcome = "fp" if spans else "tn"
        counts[outcome] += 1
        if outcome in {"fp", "fn"}:
            failures.append({"case": case.identifier, "outcome": outcome})
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "cases": len(cases),
        "counts": counts,
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "warm_mean_ms": round(mean(durations), 3) if durations else 0,
        "failures": failures,
    }


def main() -> None:
    report = evaluate(PrivacyGuard())
    report["versions"] = {name: version(name) for name in ("presidio-analyzer", "spacy", "en-core-web-sm")}
    report["model"] = MODEL_NAME
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
