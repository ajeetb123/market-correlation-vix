"""Tests for the eval runner, aggregation, summary, and the eval command."""

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from tests.fakes import FakeClient, text_msg, tool_msg
from vixagent import cli
from vixagent.agent.service import ResearchService
from vixagent.config import Settings, find_project_root, load_preregistered, load_settings
from vixagent.evals.cases import EvalCase
from vixagent.evals.graders import GraderResult
from vixagent.evals.runner import CaseRun, aggregate, render_summary, run_case, run_suite

PRE_TRAIN = {
    "group": "risk",
    "window": 21,
    "z_threshold": 2.0,
    "horizon": 10,
    "period": "train",
    "clean_only": True,
    "direction": "corr_to_vix",
}
JUDGE_PASS = '{"pass": true, "reason": "meets rubric"}'


@pytest.fixture
def service(panel: pd.DataFrame, small_settings: Settings) -> ResearchService:
    return ResearchService(small_settings, load_preregistered(load_settings()), panel)


def _case(**overrides: Any) -> EvalCase:
    base: dict[str, Any] = {
        "id": "fact-t",
        "category": "factual",
        "question": "Using the preregistered spec, what are the train hit rate and base rate?",
        "required_tools": ["run_event_study"],
        "graders": ["grounding", "numeric", "tools", "judge"],
        "expected": [
            {"reference": "event_study", "args": PRE_TRAIN, "field": "hit_rate"},
            {"reference": "event_study", "args": PRE_TRAIN, "field": "base_rate"},
        ],
        "rubric": "Pass if it states both rates.",
    }
    return EvalCase.model_validate({**base, **overrides})


def _answer(service: ResearchService, extra: str = "") -> str:
    es = service.event_study(**PRE_TRAIN)
    assert es["hit_rate"] is not None and es["base_rate"] is not None
    return f"Hit rate {es['hit_rate']:.4f} vs base rate {es['base_rate']:.4f}.{extra}"


def _agent(service: ResearchService, extra: str = "") -> FakeClient:
    return FakeClient(
        [tool_msg(("t1", "run_event_study", PRE_TRAIN)), text_msg(_answer(service, extra))]
    )


def _run(r_id: str, passed: bool | None, cat: str = "factual") -> CaseRun:
    graders = [GraderResult("numeric", passed, 1.0 if passed else 0.0, "detail here")]
    return CaseRun(r_id, cat, 1, "", [], graders, passed, 10, 5, 0.1, None)


def test_aggregate_outcomes() -> None:
    runs = [
        _run("a", True),
        _run("a", True),
        _run("b", False),
        _run("b", False),
        _run("c", True),
        _run("c", False),
        _run("d", None),
        _run("d", None),
        _run("e", True),
        _run("e", None),
    ]
    assert aggregate(runs) == {
        "a": "pass",
        "b": "fail",
        "c": "flaky",
        "d": "error",
        "e": "pass",
    }


def test_run_case_passes_with_correct_numbers(service: ResearchService, tmp_path: Path) -> None:
    judge = FakeClient([text_msg(JUDGE_PASS)])
    run = run_case(_case(), 1, service, _agent(service), judge, "fake", "fake", tmp_path)
    assert run.passed is True, [g.detail for g in run.graders]
    assert [g.name for g in run.graders] == ["grounding", "numeric", "tools", "judge"]
    assert run.input_tokens == 20
    assert len(list(tmp_path.glob("*.jsonl"))) == 1


def test_run_case_invented_number_fails_grounding(service: ResearchService, tmp_path: Path) -> None:
    judge = FakeClient([text_msg(JUDGE_PASS)])
    agent = _agent(service, " Historically 87.3% of crashes follow.")
    run = run_case(_case(), 1, service, agent, judge, "fake", "fake", tmp_path)
    assert run.passed is False
    grounding = run.graders[0]
    assert grounding.passed is False and "87.3%" in grounding.detail


def test_run_case_judge_error_is_none(service: ResearchService, tmp_path: Path) -> None:
    judge = FakeClient([text_msg("??") for _ in range(3)])
    run = run_case(_case(), 1, service, _agent(service), judge, "fake", "fake", tmp_path)
    assert run.passed is None


def test_truth_is_filled_into_rubric(service: ResearchService, tmp_path: Path) -> None:
    oos_args = {"group": "risk", "window": 21, "horizon": 10, "z_threshold": 2.0}
    case = _case(
        graders=["judge"],
        expected=[],
        required_tools=[],
        rubric="Ground truth: {truth}",
        truth={
            "compare": "gt",
            "left": {"reference": "oos", "args": oos_args, "field": "event_study_test.lift"},
            "right": {"reference": "oos", "args": oos_args, "field": "event_study_train.lift"},
        },
    )
    judge = FakeClient([text_msg(JUDGE_PASS)])
    run_case(case, 1, service, FakeClient([text_msg("No.")]), judge, "fake", "fake", tmp_path)
    prompt = judge.requests[0]["messages"][0]["content"]
    assert "{truth}" not in prompt
    assert any(word in prompt for word in ("TRUE:", "FALSE:", "UNDEFINED:"))


def test_render_summary() -> None:
    runs = [_run("a", True), _run("b", False, "trap")]
    md = render_summary(runs, aggregate(runs), {"agent_model": "m", "repeats": 1})
    assert "Overall: 1/2 pass (50.0%)" in md
    assert "| trap | 0 | 1 |" in md
    assert "**b** (fail): numeric: detail here" in md
    assert "- input: 20" in md


def test_run_suite_writes_files(service: ResearchService, tmp_path: Path) -> None:
    cases = [_case(), _case(id="fact-u")]
    agent = FakeClient(
        [
            tool_msg(("t1", "run_event_study", PRE_TRAIN)),
            text_msg(_answer(service)),
            tool_msg(("t2", "run_event_study", PRE_TRAIN)),
            text_msg(_answer(service)),
        ]
    )
    judge = FakeClient([text_msg(JUDGE_PASS), text_msg(JUDGE_PASS)])
    out = run_suite(cases, 1, tmp_path, service, agent, judge, "fake", "fake")
    data = json.loads((out / "results.json").read_text())
    assert data["outcomes"] == {"fact-t": "pass", "fact-u": "pass"}
    assert len(data["runs"]) == 2
    assert "Overall: 2/2 pass" in (out / "summary.md").read_text()


def test_eval_command(
    monkeypatch: pytest.MonkeyPatch, service: ResearchService, tmp_path: Path
) -> None:
    shutil.copytree(find_project_root() / "evals" / "cases", tmp_path / "evals" / "cases")
    monkeypatch.setattr(cli, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(cli.ResearchService, "from_cache", classmethod(lambda cls: service))
    client = FakeClient(
        [tool_msg(("t1", "run_event_study", PRE_TRAIN)), text_msg(_answer(service))]
    )
    monkeypatch.setattr(cli, "make_client", lambda: client)
    result = CliRunner().invoke(cli.app, ["eval", "--case", "fact-003"])
    assert result.exit_code == 0, result.stdout
    dirs = list((tmp_path / "evals" / "results").iterdir())
    assert len(dirs) == 1
    assert (dirs[0] / "summary.md").exists() and (dirs[0] / "results.json").exists()


def test_eval_calibrate_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient([text_msg(JUDGE_PASS) for _ in range(6)])
    monkeypatch.setattr(cli, "make_client", lambda: client)
    result = CliRunner().invoke(cli.app, ["eval", "--calibrate-judge"])
    assert result.exit_code == 1
    assert "3/6" in result.stdout
