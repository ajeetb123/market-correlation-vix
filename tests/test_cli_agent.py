"""Tests for the ask and chat commands (fake client, synthetic service)."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from tests.fakes import FakeClient, text_msg, tool_msg
from vixagent import cli
from vixagent.agent.client import MissingAPIKeyError
from vixagent.agent.service import ResearchService
from vixagent.config import Settings, load_preregistered, load_settings

runner = CliRunner()


@pytest.fixture
def wired(
    monkeypatch: pytest.MonkeyPatch, panel: pd.DataFrame, small_settings: Settings, tmp_path: Path
) -> Any:
    service = ResearchService(small_settings, load_preregistered(load_settings()), panel)
    monkeypatch.setattr(cli.ResearchService, "from_cache", classmethod(lambda cls: service))
    monkeypatch.setattr(cli, "find_project_root", lambda: tmp_path)

    def install(responses: list[Any]) -> FakeClient:
        client = FakeClient(responses)
        monkeypatch.setattr(cli, "make_client", lambda: client)
        return client

    return install


def test_ask_prints_answer(wired: Any, tmp_path: Path) -> None:
    wired([tool_msg(("t1", "describe_dataset", {})), text_msg("Fake final answer")])
    result = runner.invoke(cli.app, ["ask", "hi", "--show-tools"])
    assert result.exit_code == 0, result.stdout
    assert "Fake final answer" in result.stdout
    assert "describe_dataset" in result.stdout
    assert len(list((tmp_path / "runs").glob("*.jsonl"))) == 1


def test_ask_missing_key_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> None:
        raise MissingAPIKeyError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env.")

    monkeypatch.setattr(cli, "make_client", missing)
    result = runner.invoke(cli.app, ["ask", "hi"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY is not set" in result.stdout


def test_chat_reset_and_exit(wired: Any) -> None:
    client = wired([text_msg("First answer"), text_msg("Second answer")])
    result = runner.invoke(cli.app, ["chat"], input="hello\n/reset\nagain\n/exit\n")
    assert result.exit_code == 0, result.stdout
    assert "History cleared." in result.stdout
    assert "First answer" in result.stdout and "Second answer" in result.stdout
    assert len(client.requests[1]["messages"]) == 1


def test_chat_keeps_history(wired: Any) -> None:
    client = wired([text_msg("One"), text_msg("Two")])
    result = runner.invoke(cli.app, ["chat"], input="first\nsecond\n")
    assert result.exit_code == 0
    assert len(client.requests[1]["messages"]) == 3
