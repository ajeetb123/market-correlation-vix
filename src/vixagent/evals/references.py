"""Reference values for eval cases, computed from the live pipeline at eval time."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from vixagent.agent.service import ResearchService
from vixagent.evals.cases import RefSpec, TruthSpec

REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "event_study": lambda svc, **a: svc.event_study(**a),
    "spike_events": lambda svc, **a: svc.spike_events(**a),
    "correlation_summary": lambda svc, **a: svc.correlation_summary(**a),
    "regression": lambda svc, **a: svc.regression(**a),
    "oos": lambda svc, **a: svc.oos(**a),
    "overfitting_check": lambda svc, **a: svc.overfitting_check(),
    "dataset": lambda svc, **a: svc.describe_dataset(),
}


def get_path(obj: Any, path: str) -> Any:
    """Dot-path lookup ('coefs.z.coef'). '' returns obj. KeyError with the
    full path on failure."""
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"'{path}' not found (failed at '{part}')")
        cur = cur[part]
    return cur


def resolve(ref: RefSpec, service: ResearchService) -> Any:
    """Compute the reference value with the same service the agent's tools use.

    Why at eval time: expected values then always reflect the current data
    snapshot and code; hardcoded numbers would silently go stale.
    """
    return get_path(REGISTRY[ref.reference](service, **ref.args), ref.field)


def _fmt(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "undefined"
    return f"{x:.4g}"


def _undefined(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def resolve_truth(truth: TruthSpec, service: ResearchService) -> str:
    """Human-readable ground truth for the judge, e.g.
       gt:   'TRUE: 1.42 > 1.18'   or 'FALSE: 0.95 > 1.18 does not hold'
       sign: 'positive (0.0132)' / 'negative (-0.0071)' / 'zero'
    Values formatted with 4 significant digits. None/NaN renders as 'undefined'."""
    if truth.compare == "sign":
        assert truth.value is not None
        v = resolve(truth.value, service)
        if _undefined(v):
            return "undefined"
        if v > 0:
            return f"positive ({_fmt(v)})"
        if v < 0:
            return f"negative ({_fmt(v)})"
        return "zero"
    assert truth.left is not None and truth.right is not None
    left, right = resolve(truth.left, service), resolve(truth.right, service)
    op = ">" if truth.compare == "gt" else "<"
    expr = f"{_fmt(left)} {op} {_fmt(right)}"
    if _undefined(left) or _undefined(right):
        return f"UNDEFINED: {expr} cannot be evaluated"
    holds = left > right if truth.compare == "gt" else left < right
    return f"TRUE: {expr}" if holds else f"FALSE: {expr} does not hold"
