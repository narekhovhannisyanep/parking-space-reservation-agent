"""Local, layered privacy checks; not authentication or a complete PII/injection detector."""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from threading import RLock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

MODEL_NAME = "en_core_web_sm"
SAFE_REPLY = "Unable to process this content safely. Please retry with only parking questions or necessary reservation details. No request has been submitted, approved, or recorded."
ENTITIES = ("PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "US_SSN", "IP_ADDRESS")
_MODEL_LOCK = RLock()
_SECRET_PATTERNS = (
    r"\bsk-[a-zA-Z0-9_-]{16,}\b",
    r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    r"\b(?:api[_ -]?key|password|client[_ -]?secret|access[_ -]?token)[\"']?\s*[:=]\s*[\"']?([^\s\"',;]+)",
    r"\bBearer\s+[a-zA-Z0-9._~+/-]+=*",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?(?:-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|$)",
)
_LABELED_PRIVATE = re.compile(
    r"(?:\bfirst name|\bsurname|\bcar number(?:\s*\([^\n)]*\))?|\bvehicle identifier|\bmy name is|\bi am)"
    r"\s*[:=]?\s+([^\n;,.]+)",
    re.IGNORECASE,
)


class GuardrailError(RuntimeError):
    """Fixed-message failures never include the rejected data or detector exception."""


@dataclass(frozen=True)
class SensitiveSpan:
    start: int
    end: int
    entity: str


@cache
def _local_analyzer():
    # No imports/model loading until explicitly preparing or checking content.
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_analyzer.nlp_engine import SpacyNlpEngine
    from presidio_analyzer.predefined_recognizers import (
        CreditCardRecognizer,
        IbanRecognizer,
        IpRecognizer,
        PhoneRecognizer,
        SpacyRecognizer,
        UsSsnRecognizer,
    )

    class InstalledModelEngine(SpacyNlpEngine):
        @staticmethod
        def _download_spacy_model_if_needed(model_name: str) -> None:
            import spacy

            if not spacy.util.is_package(model_name):
                raise GuardrailError("Local privacy model is missing. Run uv sync --dev before starting.")

    # Avoid EmailRecognizer's tldextract network/cache behavior; detection uses a local pattern.
    email = PatternRecognizer(
        supported_entity="EMAIL_ADDRESS",
        patterns=[Pattern("local email", r"[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+", 0.8)],
    )
    registry = RecognizerRegistry(
        recognizers=[
            SpacyRecognizer(),
            email,
            PhoneRecognizer(),
            CreditCardRecognizer(),
            IbanRecognizer(),
            UsSsnRecognizer(),
            IpRecognizer(),
        ]
    )
    engine = InstalledModelEngine(models=[{"lang_code": "en", "model_name": MODEL_NAME}])
    return AnalyzerEngine(nlp_engine=engine, registry=registry, supported_languages=["en"])


def _detect(text: str) -> list[SensitiveSpan]:
    # Share immutable model resources, never conversations; serialize local NLP work.
    with _MODEL_LOCK:
        results = _local_analyzer().analyze(text=text, language="en", entities=list(ENTITIES), score_threshold=0.4)
    return [SensitiveSpan(item.start, item.end, item.entity_type) for item in results]


def guarded_model_config(config: RunnableConfig) -> RunnableConfig:
    """Do not stream raw internal/model tokens before the response has been checked.

    This does not hide trusted debug events or traces; those may contain private data.
    """
    return {**config, "tags": [*config.get("tags", []), "nostream"]}


class PrivacyGuard:
    """Inject a deterministic span detector in tests; real deployments use local NLP.

    Locations and dates are intentionally not blanket PII entities: they are public
    parking facts/necessary booking fields. Unknown vehicle identifiers require known
    draft values and labeled-value rules in addition to the English PERSON detector.
    """

    def __init__(self, detector: Callable[[str], Sequence[SensitiveSpan]] | None = None) -> None:
        self._detector = detector if detector is not None else _detect

    def prepare(self) -> None:
        self.spans("Parking information.")

    def spans(self, text: str, *, keep_names: bool = False, known_values: Sequence[str] = ()) -> list[SensitiveSpan]:
        if len(text) > 20000:
            raise GuardrailError("Privacy inspection limit exceeded. Use shorter text.")
        if not text:
            return []
        try:
            spans = list(self._detector(text))
            if any(
                not isinstance(span, SensitiveSpan)
                or type(span.start) is not int
                or type(span.end) is not int
                or not 0 <= span.start < span.end <= len(text)
                or not isinstance(span.entity, str)
                for span in spans
            ):
                raise GuardrailError("Invalid privacy detector result.")
        except Exception:  # noqa: BLE001 -- Fail closed at the third-party detector boundary, without logging PII.
            # Deliberately contain third-party detector failures; never propagate raw PII.
            raise GuardrailError("Local privacy checks failed. Check the installed model and retry.") from None
        # The small English model sometimes calls generated "Space B01" or "Price"
        # in a numeric price phrase a PERSON. Only these narrow public syntaxes are
        # excluded, not arbitrary names, addresses, dates, or whole records.
        amounts = [
            (match.start(), match.end())
            for match in re.finditer(r"\b(?:Prices?\s*:?\s*)?(?:EUR|USD|AMD|GBP)\s+\d+(?:\.\d+)?\b", text)
        ]
        spans = [
            span
            for span in spans
            if not (
                span.entity == "PERSON"
                and (
                    re.fullmatch(r"Space [A-Z]\d{1,2}(?:-L\d+)?", text[span.start : span.end])
                    or re.fullmatch(r"L\d+-[A-Z]\d{2}", text[span.start : span.end])
                    or any(start <= span.start < span.end <= end for start, end in amounts)
                    or (
                        text[span.start : span.end] in {"Price", "Prices"}
                        and re.match(r"Prices?\s*:?\s*(?:(?:EUR|USD|AMD|GBP)\s+)?\d", text[span.start :])
                    )
                )
            )
        ]
        if keep_names:
            spans = [span for span in spans if span.entity != "PERSON"]
        for pattern in _SECRET_PATTERNS:
            spans.extend(
                SensitiveSpan(match.start(), match.end(), "SECRET")
                for match in re.finditer(pattern, text, re.IGNORECASE)
            )
        if not keep_names:
            for value in known_values:
                if value:
                    spans.extend(
                        SensitiveSpan(match.start(), match.end(), "RESERVATION_DETAIL")
                        for match in re.finditer(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.IGNORECASE)
                    )
            spans.extend(
                SensitiveSpan(match.start(1), match.end(1), "RESERVATION_DETAIL")
                for match in _LABELED_PRIVATE.finditer(text)
            )
        return spans

    def redact(self, text: str, *, keep_names: bool = False, known_values: Sequence[str] = ()) -> str:
        ranges = sorted(
            (span.start, span.end) for span in self.spans(text, keep_names=keep_names, known_values=known_values)
        )
        merged: list[tuple[int, int]] = []
        for start, end in ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        for start, end in reversed(merged):
            # Spaces preserve offsets and cannot become an artificial extracted identifier.
            text = text[:start] + " " * (end - start) + text[end:]
        return text

    def check_public(self, value: object) -> None:
        """Reject, do not silently modify, public records (including metadata/JSON keys)."""
        if isinstance(value, str):
            spans = self.spans("Public parking data: " + value)
            # Numeric-suffixed technical identifiers are public inventory/location
            # labels, not names. Still reject email, credentials and financial PII.
            if re.fullmatch(r"[a-z][a-z_-]*-\d+", value):
                spans = [span for span in spans if span.entity != "PERSON"]
            if spans:
                raise GuardrailError("Public parking data contains detected sensitive information. Review the source.")
        elif isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(key, str) and key.lower().replace("-", "_").replace(" ", "_") in {
                    "first_name",
                    "surname",
                    "car_number",
                    "password",
                    "api_key",
                    "authorization",
                    "access_token",
                    "client_secret",
                }:
                    raise GuardrailError("Private field names are not allowed in public parking data.")
                # Schema-owned key labels are fixed, not user content. Inspect
                # their values and every arbitrary key, including nested keys.
                if key not in {"space_id", "location_id", "label", "source", "record_type", "schema_version"}:
                    self.check_public(key)
                self.check_public(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.check_public(item)

    def check_document(self, document: Document) -> None:
        self.check_public(document.metadata)
        prefix = (
            f"Space {document.metadata['label']} ({document.metadata['space_id']}) at {document.metadata['location_id']}. "
            "Accepts any vehicle under this project's assumption. "
        )
        # Strip only the exact code-generated prefix, after checking its metadata.
        # Replaced/operator-supplied prose is inspected in full.
        self.check_public(document.page_content.removeprefix(prefix))

    @staticmethod
    def text_content(message: BaseMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        texts = []
        for block in message.content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
            else:
                raise GuardrailError("Privacy checks support text content only. Please provide a text message.")
        return "\n".join(texts)

    def public_messages(self, messages: Sequence[AnyMessage], known_values: Sequence[str]) -> list[AnyMessage]:
        """Copies contain text only, never private names/kwargs/tool payloads supplied by a caller."""
        copies: list[AnyMessage] = []
        for message in messages:
            if isinstance(message, (HumanMessage, AIMessage)):
                text = self.redact(self.text_content(message), known_values=known_values)
                kind = HumanMessage if isinstance(message, HumanMessage) else AIMessage
                copies.append(kind(content=text, id=message.id))
        return copies

    def response(self, response: BaseMessage, known_values: Sequence[str]) -> AIMessage:
        if not isinstance(response, AIMessage) or response.tool_calls or response.invalid_tool_calls:
            return AIMessage(SAFE_REPLY)
        try:
            text = self.redact(self.text_content(response), known_values=known_values).strip()
        except GuardrailError:
            text = SAFE_REPLY
        return AIMessage(text or SAFE_REPLY, id=response.id)
