"""Tests for the agent loop, prompt, client factory, and transcripts (fake client only)."""

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.fakes import FakeClient, text_msg, tool_msg
from vixagent.agent import client as client_mod
from vixagent.agent.client import MissingAPIKeyError, agent_model, make_client
from vixagent.agent.loop import LIMIT_MESSAGE, MAX_ITERATIONS, run_agent
from vixagent.agent.prompts import build_system_prompt
from vixagent.agent.service import ResearchService
from vixagent.agent.tools import TOOL_SCHEMAS
from vixagent.agent.transcript import TranscriptWriter, redact, slugify
from vixagent.config import Settings, load_preregistered, load_settings

HEADLINE_ES = {
    "group": "risk",
    "window": 21,
    "z_threshold": 2.0,
    "horizon": 10,
    "period": "test",
    "clean_only": True,
    "direction": "corr_to_vix",
}


@pytest.fixture
def service(panel: pd.DataFrame, small_settings: Settings) -> ResearchService:
    return ResearchService(small_settings, load_preregistered(load_settings()), panel)


def _run(service: ResearchService, responses: list[Any], **kwargs: Any) -> tuple[Any, FakeClient]:
    client = FakeClient(responses)
    result = run_agent(
        "What is the headline?", client=client, service=service, model="fake", **kwargs
    )
    return result, client


def test_single_tool_call_then_answer(service: ResearchService) -> None:
    result, client = _run(service, [tool_msg(("t1", "describe_dataset", {})), text_msg("Done")])
    assert result.final_text == "Done"
    assert result.stop_reason == "end_turn"
    assert len(result.tool_calls) == 1 and result.tool_calls[0].name == "describe_dataset"
    assert result.iterations == 2
    msgs = client.requests[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"][0].type == "tool_use"
    results = msgs[2]["content"]
    assert len(results) == 1
    assert results[0]["type"] == "tool_result"
    assert results[0]["tool_use_id"] == "t1"
    assert results[0]["is_error"] is False
    assert json.loads(results[0]["content"])["data_source"]


def test_two_tool_calls_answered_in_one_message(service: ResearchService) -> None:
    responses = [
        tool_msg(
            ("a", "get_preregistered_spec", {}),
            ("b", "run_event_study", HEADLINE_ES),
            preface="Checking.",
        ),
        text_msg("Answer"),
    ]
    result, client = _run(service, responses)
    tool_results = client.requests[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in tool_results] == ["a", "b"]
    assert [c.name for c in result.tool_calls] == ["get_preregistered_spec", "run_event_study"]
    assert result.tool_calls[1].input == HEADLINE_ES


def test_invalid_args_become_error_result(service: ResearchService) -> None:
    bad = {**HEADLINE_ES, "window": 5}
    result, client = _run(service, [tool_msg(("x", "run_event_study", bad)), text_msg("Fixed")])
    tool_result = client.requests[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert result.tool_calls[0].is_error is True
    assert result.final_text == "Fixed"


def test_iteration_limit(service: ResearchService) -> None:
    responses = [tool_msg((f"t{i}", "describe_dataset", {})) for i in range(MAX_ITERATIONS)]
    result, _ = _run(service, responses)
    assert result.stop_reason == "iteration_limit"
    assert result.final_text == LIMIT_MESSAGE
    assert result.iterations == MAX_ITERATIONS
    assert result.messages[-1]["role"] == "assistant"


def test_max_tokens_note(service: ResearchService) -> None:
    result, _ = _run(service, [text_msg("Partial", stop_reason="max_tokens")])
    assert result.final_text.endswith("[Answer truncated: max_tokens reached.]")
    assert result.stop_reason == "max_tokens"


def test_usage_sums(service: ResearchService) -> None:
    responses = [tool_msg(("t1", "describe_dataset", {})), text_msg("Done")]
    result, _ = _run(service, responses)
    assert result.usage == {"input_tokens": 20, "output_tokens": 10}


def test_history_is_prepended(service: ResearchService) -> None:
    history = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": [{"type": "text", "text": "Earlier answer"}]},
    ]
    _, client = _run(service, [text_msg("Now")], history=history)
    msgs = client.requests[0]["messages"]
    assert len(msgs) == 3
    assert msgs[0]["content"] == "Earlier question"


def test_requests_carry_tools_and_system(service: ResearchService) -> None:
    responses = [tool_msg(("t1", "describe_dataset", {})), text_msg("Done")]
    _, client = _run(service, responses)
    for req in client.requests:
        assert req["tools"] == TOOL_SCHEMAS
        assert req["system"] == build_system_prompt(service.settings)
        assert req["model"] == "fake"


def test_transcript_has_no_key(
    service: ResearchService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-SECRET123")
    writer = TranscriptWriter(tmp_path, "Leak test sk-test-SECRET123")
    responses = [
        tool_msg(("t1", "describe_dataset", {})),
        text_msg("The key is sk-test-SECRET123"),
    ]
    _run(service, responses, transcript=writer)
    text = writer.path.read_text()
    assert "SECRET123" not in text
    events = [json.loads(line)["event"] for line in text.splitlines()]
    assert events == [
        "user",
        "assistant",
        "tool_call",
        "tool_result",
        "assistant",
        "final",
        "usage",
    ]


def test_system_prompt_is_filled() -> None:
    s = load_settings()
    prompt = build_system_prompt(s)
    assert not re.search(r"\{[a-z_]+\}", prompt)
    assert s.data.start.isoformat() in prompt and s.data.snapshot_end.isoformat() in prompt
    for tickers in s.universe.values():
        for t in tickers:
            assert t in prompt
    assert "risk (SPY, QQQ, IWM, EFA, EEM, HYG) and all (" in prompt


def test_slugify() -> None:
    assert slugify("What is the headline result?") == "what-is-the-headline-result"
    assert slugify("  A/B -- C!! ") == "a-b-c"
    assert len(slugify("x" * 100)) == 40
    assert slugify("???") == "run"


def test_transcript_lines_are_json(tmp_path: Path) -> None:
    w = TranscriptWriter(tmp_path / "runs", "hello world")
    w.write("user", "hello world")
    w.write("final", {"text": "hi"})
    assert w.path.name.endswith("-hello-world.jsonl")
    lines = w.path.read_text().splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["user", "final"]


def test_redact_ignores_short_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "abc")
    assert redact("abc def") == "abc def"


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(client_mod, "load_dotenv", lambda: None)
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY is not set"):
        make_client()


def test_model_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIX_AGENT_MODEL", "some-model")
    assert agent_model() == "some-model"
