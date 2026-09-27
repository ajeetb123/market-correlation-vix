"""Deterministic graders: grounding, numeric, date, and tools."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from vixagent.agent.loop import ToolCallRecord
from vixagent.config import Preregistered, Settings


@dataclass
class GraderResult:
    name: str
    passed: bool | None  # None = error (judge only)
    score: float
    detail: str


@dataclass(frozen=True)
class Num:
    value: float  # as written (percent NOT divided)
    is_percent: bool
    decimals: int  # digits after the decimal point as written
    raw: str

    @property
    def sig_digits(self) -> int:
        """Significant digits as written, ignoring sign, commas, and leading zeros."""
        digits = re.sub(r"\D", "", self.raw)
        return len(digits.lstrip("0")) or 1


DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.)]\s")
NUM_RE = re.compile(r"(?<![\w.])[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
UNAVAILABLE_PHRASES = (
    "not available",
    "unavailable",
    "undefined",
    "no events",
    "insufficient",
    "n/a",
)


def extract_numbers(text: str) -> list[Num]:
    """1. Remove dates (DATE_RE) and list markers (LIST_MARKER_RE).
    2. Find NUM_RE matches; strip commas; unicode minus to '-'.
    3. Skip integers 1900..2100 with no decimals and no '%' (years).
    """
    cleaned = LIST_MARKER_RE.sub(" ", DATE_RE.sub(" ", text))
    out: list[Num] = []
    for m in NUM_RE.finditer(cleaned):
        raw = m.group(0)
        is_percent = raw.endswith("%")
        body = raw.rstrip("%").replace(",", "").replace("−", "-")
        decimals = len(body.split(".")[1]) if "." in body else 0
        value = float(body)
        if not is_percent and decimals == 0 and 1900 <= value <= 2100:
            continue
        out.append(Num(value, is_percent, decimals, raw))
    return out


def number_matches(a: Num, b: float) -> bool:
    """True if `a` (as written) is consistent with source value `b`.

    Compares in the units the answer used:
      - if a.is_percent: target = b * 100 (source fraction) and also b itself
        (source already in percent)
      - else: target = b, and also b * 100 (answer wrote '34.5' for 0.345)
    Consistent if |a.value - target| <= max(0.5 * 10**(-a.decimals) + 1e-9,
                                             0.01 * abs(target)).
    Why: an answer rounding 1.3456 to '1.3' is faithful, not invented."""
    for target in (b * 100, b) if a.is_percent else (b, b * 100):
        tol = max(0.5 * 10 ** (-a.decimals) + 1e-9, 0.01 * abs(target))
        if abs(a.value - target) <= tol:
            return True
    return False


def _numbers_in(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, int | float):
        if math.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(n.value / 100 if n.is_percent else n.value for n in extract_numbers(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _numbers_in(v, out)


def collect_sources(
    tool_calls: list[ToolCallRecord], question: str, settings: Settings, prereg: Preregistered
) -> list[float]:
    """Every number in: tool outputs and inputs (recursively, including numbers
    inside string values via extract_numbers), the question text, and
    settings/prereg model_dump(). Booleans excluded."""
    out: list[float] = []
    for call in tool_calls:
        _numbers_in(call.output, out)
        _numbers_in(call.input, out)
    _numbers_in(question, out)
    _numbers_in(settings.model_dump(mode="json"), out)
    _numbers_in(prereg.model_dump(mode="json"), out)
    return out


def grade_grounding(answer: str, sources: list[float]) -> GraderResult:
    """Pass if every extracted number matches some source (or there are none).
    score = grounded / total; detail lists ungrounded raw strings.

    Why: rule 1 of the system prompt says every number must come from a tool.
    This checks that mechanically instead of trusting the model."""
    nums = extract_numbers(answer)
    if not nums:
        return GraderResult("grounding", True, 1.0, "no numbers in answer")
    ungrounded = [n.raw for n in nums if not any(number_matches(n, s) for s in sources)]
    score = (len(nums) - len(ungrounded)) / len(nums)
    detail = f"ungrounded: {ungrounded}" if ungrounded else f"all {len(nums)} numbers grounded"
    return GraderResult("grounding", not ungrounded, score, detail)


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def grade_numeric(answer: str, expected_values: list[Any]) -> GraderResult:
    """For each expected value:
      - None/NaN: pass if the answer contains (case-insensitive) one of
        'not available', 'unavailable', 'undefined', 'no events',
        'insufficient', 'n/a'.
      - int: some extracted number equals it exactly.
      - float: some extracted number matches it (number_matches) AND shows at
        least 2 significant digits (so 'about 1' does not match 1.3456).
    All must pass. score = fraction matched."""
    nums = extract_numbers(answer)
    lower = answer.lower()
    missed: list[str] = []
    for v in expected_values:
        if _is_missing(v):
            ok = any(p in lower for p in UNAVAILABLE_PHRASES)
        elif isinstance(v, int) and not isinstance(v, bool):
            ok = any(not n.is_percent and n.value == v for n in nums)
        else:
            ok = any(n.sig_digits >= 2 and number_matches(n, float(v)) for n in nums)
        if not ok:
            missed.append(repr(v))
    total = len(expected_values) or 1
    score = (len(expected_values) - len(missed)) / total
    detail = f"missing expected: {', '.join(missed)}" if missed else "all expected values present"
    return GraderResult("numeric", not missed, score, detail)


def grade_date(answer: str, expected_dates: list[str]) -> GraderResult:
    """Each expected YYYY-MM-DD appears verbatim."""
    missing = [d for d in expected_dates if str(d) not in answer]
    score = (len(expected_dates) - len(missing)) / (len(expected_dates) or 1)
    detail = f"missing dates: {missing}" if missing else "all dates present"
    return GraderResult("date", not missing, score, detail)


def grade_tools(called: list[str], required: list[str]) -> GraderResult:
    """Each required tool called at least once. detail lists missing."""
    missing = [t for t in required if t not in called]
    score = (len(required) - len(missing)) / (len(required) or 1)
    detail = f"missing tools: {missing}" if missing else "all required tools called"
    return GraderResult("tools", not missing, score, detail)
