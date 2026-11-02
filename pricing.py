"""Bounded English USD/hour preferences. No databases, models, or import side effects."""

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

BUDGET_CLARIFICATION = (
    "Please clarify your hourly USD price limit, for example 'up to $8/hour' or "
    "'between $5 and $10/hour', or send /clear-budget. No new options or submission "
    "are allowed until the price preference is resolved."
)


class BudgetError(ValueError):
    """Fixed safe clarification; never echo raw extraction or user data."""

    def __init__(self) -> None:
        super().__init__(BUDGET_CLARIFICATION)


class HourlyBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, revalidate_instances="always")

    minimum_cents: int | None = Field(default=None, ge=0, le=1_000_000_000)
    maximum_cents: int | None = Field(default=None, ge=0, le=1_000_000_000)
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True

    @model_validator(mode="after")
    def validate_bounds(self) -> "HourlyBudget":
        if self.minimum_cents is None and self.maximum_cents is None:
            raise ValueError("A budget requires a numeric bound.")
        if (
            self.minimum_cents is not None
            and self.maximum_cents is not None
            and (
                self.minimum_cents > self.maximum_cents
                or (
                    self.minimum_cents == self.maximum_cents and not (self.minimum_inclusive and self.maximum_inclusive)
                )
            )
        ):
            raise ValueError("Budget range must not be reversed or empty.")
        return self

    def clauses(self) -> list[dict]:
        result = []
        if self.minimum_cents is not None:
            result.append({"hourly_price_cents": {"$gte" if self.minimum_inclusive else "$gt": self.minimum_cents}})
        if self.maximum_cents is not None:
            result.append({"hourly_price_cents": {"$lte" if self.maximum_inclusive else "$lt": self.maximum_cents}})
        return result

    def accepts(self, cents: int) -> bool:
        validate_rate(cents)
        return (
            self.minimum_cents is None
            or cents > self.minimum_cents
            or (self.minimum_inclusive and cents == self.minimum_cents)
        ) and (
            self.maximum_cents is None
            or cents < self.maximum_cents
            or (self.maximum_inclusive and cents == self.maximum_cents)
        )

    def summary(self) -> str:
        parts = []
        if self.minimum_cents is not None:
            parts.append(("at least " if self.minimum_inclusive else "more than ") + money(self.minimum_cents))
        if self.maximum_cents is not None:
            parts.append(("up to " if self.maximum_inclusive else "less than ") + money(self.maximum_cents))
        return "Hourly price filter: " + " and ".join(parts) + "/hour (USD)."


def validate_rate(cents: int) -> None:
    if type(cents) is not int or not 100 <= cents <= 2000:
        raise ValueError("Mock hourly prices must be integer cents between 100 and 2000 (USD).")


def money(cents: int) -> str:
    return f"${cents // 100}.{cents % 100:02d}"


WORDS = dict(
    enumerate(
        (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        )
    )
)
WORD_VALUES = {word: number for number, word in WORDS.items()}
NUMBER = r"(?:[0-9]+(?:\.[0-9]+)?|" + "|".join(WORD_VALUES) + r")"
AMOUNT = r"\$?\s*" + NUMBER + r"\s*(?:USD\b|dollars?\b)?"
UNIT = r"\s*(?:/\s*(?:hours?|hrs?|h)\b|(?:per|an|a)\s+hour\b)"
OPERATOR = (
    r"(?:under|below|less than|up to|at most|no more than|maximum|max|budget(?:\s+is|\s+of|:)?)(?:\s+hourly price)?"
)
RANGE = re.compile(r"\bbetween\s+(?P<low>" + AMOUNT + r")\s+and\s+(?P<high>" + AMOUNT + r")" + UNIT, re.IGNORECASE)
DASH_RANGE = re.compile(
    r"(?<![\w.$-])(?P<low>" + AMOUNT + r")\s*[-–]\s*(?P<high>" + AMOUNT + r")" + UNIT, re.IGNORECASE
)
MAXIMUM = re.compile(r"\b(?P<operator>" + OPERATOR + r")\s+(?P<high>" + AMOUNT + r")" + UNIT, re.IGNORECASE)
CUES = re.compile(
    r"[$€£]|\b(?:USD|EUR|GBP|dollars?|budget|cheap|cheaper|afford|expensive)\b|\bprice\s+(?:limit|range|cap)\b",
    re.IGNORECASE,
)
CLEAR = re.compile(
    r"(?:/clear-budget|(?:clear|remove) (?:my |the )?(?:budget|price limit)|no (?:budget|price) limit)[.!]?",
    re.IGNORECASE,
)


def budget_cue(text: str) -> bool:
    if re.search(r"[$€£]|\b(?:budget|cheap|cheaper|afford)\b|\bprice\s+(?:limit|range|cap)\b", text, re.IGNORECASE):
        return True
    if re.search(r"\bexpensive\b", text, re.IGNORECASE) and not re.search(r"\bhow expensive\b", text, re.IGNORECASE):
        return True
    return bool(
        re.search(r"\b(?:USD|EUR|GBP|dollars?)\b", text, re.IGNORECASE)
        and re.search(r"(?<!\w)" + NUMBER + r"(?!\w)", text, re.IGNORECASE)
    )


def clears_budget(text: str) -> bool:
    return CLEAR.fullmatch(text.strip()) is not None


def _cents(amount: str) -> int:
    value = re.sub(r"[$\s]|USD|dollars?", "", amount, flags=re.IGNORECASE).lower()
    if value in WORD_VALUES:
        return WORD_VALUES[value] * 100
    whole, _, fraction = value.partition(".")
    # Inspect the original digits BEFORE arithmetic. Decimal multiplication uses
    # a precision context and can round a very long non-cent amount to whole cents.
    if len(fraction.rstrip("0")) > 2:
        raise BudgetError()
    cents = int(whole) * 100 + int((fraction + "00")[:2])
    if not 0 <= cents <= 1_000_000_000:
        raise BudgetError()
    return cents


def resolve_budget(text: str, proposal: str | None = None) -> HourlyBudget | None:
    """Independently parse one explicit phrase; model proposals must be verbatim and complete.

    This is a documented bounded grammar, not universal language understanding.
    Unsupported money/budget cues fail closed instead of removing an active constraint.
    """
    if proposal is not None and (not proposal or proposal not in text or len(proposal) > 200):
        raise BudgetError()
    if not budget_cue(text) and proposal is None:
        return None
    if len(text) > 2000 or re.search(
        r"\bnot\s+(?:under|below|up to|between)|\b(?:ignore|disregard|except|instead)\b", text, re.IGNORECASE
    ):
        raise BudgetError()
    matches = [match for pattern in (RANGE, DASH_RANGE, MAXIMUM) for match in pattern.finditer(text)]
    if len(matches) != 1:
        raise BudgetError()
    match = matches[0]
    phrase = match[0]
    if not re.search(r"\$|\b(?:USD|dollars?)\b", phrase, re.IGNORECASE):
        raise BudgetError()
    # No second currency, alternative budget, total price, or fuzzy modifier may be ignored.
    remainder = text[: match.start()] + " " + text[match.end() :]
    remainder = re.sub(r"\b(?:my\s+)?budget\s*(?:is|of|:)?\s*$", "", remainder, flags=re.IGNORECASE)
    if CUES.search(remainder) or re.search(
        r"\b(?:total|approximately|about|roughly|around)\b", remainder, re.IGNORECASE
    ):
        raise BudgetError()
    if proposal is not None and proposal.strip().casefold() != phrase.strip().casefold():
        raise BudgetError()
    values = match.groupdict()
    try:
        return HourlyBudget(
            minimum_cents=_cents(values["low"]) if values.get("low") is not None else None,
            maximum_cents=_cents(values["high"]),
            maximum_inclusive=values.get("operator", "").lower() not in {"under", "below", "less than"},
        )
    except ValueError:
        raise BudgetError() from None
