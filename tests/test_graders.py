"""Tests for the deterministic eval graders."""

import pytest

from vixagent.agent.loop import ToolCallRecord
from vixagent.config import load_preregistered, load_settings
from vixagent.evals.graders import (
    Num,
    collect_sources,
    extract_numbers,
    grade_date,
    grade_grounding,
    grade_numeric,
    grade_tools,
    number_matches,
)


def _num(raw: str) -> Num:
    (n,) = extract_numbers(raw)
    return n


def test_extract_numbers_spec_sentence() -> None:
    text = "Lift was **1.42** (p = 0.013) over 2019-01-02 to 2026-06-30; 5,000 permutations; 34.5%."
    nums = extract_numbers(text)
    assert [n.value for n in nums] == [1.42, 0.013, 5000, 34.5]
    assert [n.is_percent for n in nums] == [False, False, False, True]
    assert [n.decimals for n in nums] == [2, 3, 0, 1]


def test_list_markers_ignored() -> None:
    assert [n.value for n in extract_numbers("1. First\n2. Second 0.4")] == [0.4]


def test_years_skipped() -> None:
    assert extract_numbers("since 2008") == []


def test_negative_and_unicode_minus() -> None:
    assert [n.value for n in extract_numbers("coef -0.012 and −0.5")] == [-0.012, -0.5]


@pytest.mark.parametrize(
    ("raw", "source", "expected"),
    [
        ("1.3", 1.3456, True),
        ("1.4", 1.3456, False),
        ("34.5%", 0.345, True),
        ("42%", 0.4167, True),
        ("12", 12, True),
        ("12", 13, False),
        ("34.5", 0.345, True),
        ("29%", 29.0, True),
    ],
)
def test_number_matches(raw: str, source: float, expected: bool) -> None:
    assert number_matches(_num(raw), source) is expected


def test_grounding_passes_when_all_numbers_present() -> None:
    sources = [0.2903, 0.1712, 1.6957, 31.0]
    result = grade_grounding("Hit rate 29.0% vs base 17.1%, lift 1.70 on 31 events.", sources)
    assert result.passed is True
    assert result.score == 1.0


def test_grounding_fails_on_invented_number() -> None:
    sources = [0.2903, 0.1712]
    result = grade_grounding("Hit rate 29.0% vs base 17.1%, and 87.3% of crashes.", sources)
    assert result.passed is False
    assert "87.3%" in result.detail
    assert result.score == pytest.approx(2 / 3)


def test_grounding_no_numbers() -> None:
    assert grade_grounding("Bitcoin is not in the dataset.", []).passed is True


def test_collect_sources_includes_strings_inputs_and_config() -> None:
    s = load_settings()
    calls = [
        ToolCallRecord(
            "get_methodology",
            {"topic": "zscore", "window": 63},
            {"text": "uses 252 days"},
            False,
            1,
        )
    ]
    sources = collect_sources(calls, "What about z >= 2.5?", s, load_preregistered(s))
    for value in (63, 252, 2.5, 1.3, 5000, 0.05, 21):
        assert value in sources
    assert True not in [type(x) is bool for x in sources]


def test_numeric_requires_two_significant_digits() -> None:
    assert grade_numeric("The lift was about 1.", [1.3456]).passed is False
    assert grade_numeric("The lift was 1.35.", [1.3456]).passed is True


def test_numeric_percent_and_integer() -> None:
    assert grade_numeric("Hit rate 29.0% with 31 events.", [0.2903, 31]).passed is True
    assert grade_numeric("There were 30 events.", [31]).passed is False


def test_numeric_none_needs_unavailable_phrase() -> None:
    assert grade_numeric("There were no events, so the lift is undefined.", [None]).passed
    assert grade_numeric("The lift is 1.2.", [None]).passed is False
    assert grade_numeric("Hit rate is n/a.", [float("nan")]).passed is True


def test_numeric_score_partial() -> None:
    result = grade_numeric("Lift 1.35.", [1.3456, 0.0123])
    assert result.passed is False
    assert result.score == 0.5


def test_date_grader() -> None:
    assert grade_date("Peak on 2020-03-16.", ["2020-03-16"]).passed is True
    assert grade_date("Peak in March 2020.", ["2020-03-16"]).passed is False


def test_tools_grader() -> None:
    assert grade_tools(["run_event_study", "describe_dataset"], ["run_event_study"]).passed
    result = grade_tools(["describe_dataset"], ["run_event_study"])
    assert result.passed is False
    assert "run_event_study" in result.detail
