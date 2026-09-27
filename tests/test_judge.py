"""Tests for the LLM judge (fake client only)."""

from pathlib import Path

import pytest

from tests.fakes import FakeClient, text_msg
from vixagent.config import find_project_root
from vixagent.evals.judge import JUDGE_PROMPT, calibrate, parse_judge_json, run_judge


@pytest.mark.parametrize(
    "text",
    [
        '{"pass": true, "reason": "ok"}',
        '```json\n{"pass": true, "reason": "ok"}\n```',
        'Here is my grade: {"pass": true, "reason": "ok"} Hope that helps.',
        'Note {not json} then {"pass": true, "reason": "ok"}',
    ],
)
def test_parse_variants(text: str) -> None:
    assert parse_judge_json(text) == {"pass": True, "reason": "ok"}


def test_parse_rejects_missing_or_non_bool_pass() -> None:
    assert parse_judge_json('{"reason": "no verdict"}') is None
    assert parse_judge_json('{"pass": "yes"}') is None
    assert parse_judge_json("no json here") is None


def _judge(client: FakeClient) -> object:
    return run_judge(client, "fake", "Q?", "Rubric.", [], "Answer.")


def test_invalid_three_times_is_error() -> None:
    client = FakeClient([text_msg("nope"), text_msg("still nope"), text_msg("nope again")])
    verdict = _judge(client)
    assert verdict.passed is None  # type: ignore[attr-defined]
    assert len(client.requests) == 3


def test_retry_then_valid() -> None:
    client = FakeClient([text_msg("garbage"), text_msg('{"pass": false, "reason": "missed 2"}')])
    verdict = _judge(client)
    assert verdict.passed is False  # type: ignore[attr-defined]
    assert verdict.reason == "missed 2"  # type: ignore[attr-defined]
    assert len(client.requests) == 2
    assert client.requests[0]["max_tokens"] == 400


def test_prompt_contains_inputs() -> None:
    client = FakeClient([text_msg('{"pass": true, "reason": "ok"}')])
    _judge(client)
    prompt = client.requests[0]["messages"][0]["content"]
    assert "QUESTION:\nQ?" in prompt and "RUBRIC:\nRubric." in prompt
    assert '{"pass": true or false' in JUDGE_PROMPT.format(
        question="", rubric="", tool_calls="", answer=""
    )


def test_calibrate_with_always_pass_judge() -> None:
    path: Path = find_project_root() / "evals" / "judge_calibration.yaml"
    client = FakeClient([text_msg('{"pass": true, "reason": "fine"}') for _ in range(6)])
    agree, total, disagreements = calibrate(client, "fake", path)
    assert (agree, total) == (3, 6)
    assert len(disagreements) == 3
