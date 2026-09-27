"""Tests for eval case loading, references, and truth expressions."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from vixagent.agent.service import ResearchService
from vixagent.config import Settings, find_project_root, load_preregistered, load_settings
from vixagent.evals.cases import EvalCase, RefSpec, TruthSpec, load_cases
from vixagent.evals.references import get_path, resolve, resolve_truth

VALID: dict[str, Any] = {
    "id": "fact-x",
    "category": "factual",
    "question": "How many?",
    "required_tools": ["get_spike_events"],
    "graders": ["grounding", "numeric", "tools"],
    "expected": [
        {
            "reference": "spike_events",
            "args": {"kind": "vix", "period": "full"},
            "field": "n_events",
        }
    ],
}


def _write(tmp_path: Path, *cases: dict[str, Any]) -> Path:
    for i, c in enumerate(cases):
        (tmp_path / f"case-{i}.yaml").write_text(yaml.safe_dump(c))
    return tmp_path


def test_valid_case_loads(tmp_path: Path) -> None:
    cases = load_cases(_write(tmp_path, VALID))
    assert [c.id for c in cases] == ["fact-x"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"graders": ["grounding", "vibes"]}, "vibes"),
        ({"graders": ["judge"], "rubric": None}, "rubric"),
        ({"graders": ["judge"], "rubric": "Truth: {truth}"}, "truth"),
        ({"expected": [{"reference": "horoscope", "field": "x"}]}, "horoscope"),
        ({"required_tools": ["crystal_ball"]}, "crystal_ball"),
        ({"graders": ["numeric"], "expected": []}, "expected"),
    ],
)
def test_invalid_cases_rejected(tmp_path: Path, changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_cases(_write(tmp_path, {**VALID, **changes}))


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(_write(tmp_path, VALID, VALID))


def test_all_problems_reported_together(tmp_path: Path) -> None:
    bad_tool = {**VALID, "id": "a", "required_tools": ["crystal_ball"]}
    bad_ref = {**VALID, "id": "b", "expected": [{"reference": "horoscope"}]}
    with pytest.raises(ValueError) as info:
        load_cases(_write(tmp_path, bad_tool, bad_ref))
    assert "crystal_ball" in str(info.value) and "horoscope" in str(info.value)


def test_truth_spec_validation() -> None:
    with pytest.raises(ValueError):
        TruthSpec.model_validate({"compare": "gt", "left": {"reference": "oos"}})
    with pytest.raises(ValueError):
        TruthSpec.model_validate({"compare": "sign"})


def test_real_cases_load() -> None:
    cases = load_cases(find_project_root() / "evals" / "cases")
    assert len(cases) == 24
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.category] = counts.get(c.category, 0) + 1
    assert counts == {
        "factual": 6,
        "methodology": 4,
        "pushback": 6,
        "trap": 4,
        "out_of_scope": 4,
    }


def test_real_case_references_resolve(panel: pd.DataFrame, small_settings: Settings) -> None:
    service = ResearchService(small_settings, load_preregistered(load_settings()), panel)
    cases: list[EvalCase] = load_cases(find_project_root() / "evals" / "cases")
    for case in cases:
        for ref in case.expected:
            resolve(ref, service)
        if case.truth:
            assert resolve_truth(case.truth, service)


def test_get_path() -> None:
    obj = {"coefs": {"z": {"coef": 0.5}}, "n": 3}
    assert get_path(obj, "coefs.z.coef") == 0.5
    assert get_path(obj, "") is obj
    with pytest.raises(KeyError, match="coefs.x.coef"):
        get_path(obj, "coefs.x.coef")


class FakeService:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def oos(self, **_: Any) -> dict[str, Any]:
        return self.values


def _ref(field: str) -> RefSpec:
    return RefSpec(reference="oos", args={}, field=field)


def test_resolve_truth_expressions() -> None:
    svc: Any = FakeService({"a": 1.42, "b": 1.18, "neg": -0.0071, "zero": 0.0, "none": None})
    gt = TruthSpec(compare="gt", left=_ref("a"), right=_ref("b"))
    assert resolve_truth(gt, svc) == "TRUE: 1.42 > 1.18"
    lt = TruthSpec(compare="lt", left=_ref("a"), right=_ref("b"))
    assert resolve_truth(lt, svc) == "FALSE: 1.42 < 1.18 does not hold"
    assert resolve_truth(TruthSpec(compare="sign", value=_ref("a")), svc) == "positive (1.42)"
    assert resolve_truth(TruthSpec(compare="sign", value=_ref("neg")), svc) == "negative (-0.0071)"
    assert resolve_truth(TruthSpec(compare="sign", value=_ref("zero")), svc) == "zero"
    assert resolve_truth(TruthSpec(compare="sign", value=_ref("none")), svc) == "undefined"
    missing = TruthSpec(compare="gt", left=_ref("none"), right=_ref("b"))
    assert resolve_truth(missing, svc).startswith("UNDEFINED")
