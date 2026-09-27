"""Tests for CLI commands (no network)."""

from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from vixagent import cli
from vixagent.config import Settings
from vixagent.data.fetch import DataError

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_pull_prints_summary(
    monkeypatch: pytest.MonkeyPatch, panel: pd.DataFrame, small_settings: Settings
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: small_settings)
    monkeypatch.setattr(cli, "load_or_fetch", lambda *a, **k: panel)
    result = runner.invoke(cli.app, ["pull"])
    assert result.exit_code == 0
    assert "risk" in result.stdout
    assert "rows dropped" in result.stdout


def test_pull_data_error_exits_1(monkeypatch: pytest.MonkeyPatch, small_settings: Settings) -> None:
    def boom(*_: Any, **__: Any) -> pd.DataFrame:
        raise DataError("no data")

    monkeypatch.setattr(cli, "load_settings", lambda: small_settings)
    monkeypatch.setattr(cli, "load_or_fetch", boom)
    result = runner.invoke(cli.app, ["pull"])
    assert result.exit_code == 1
    assert "no data" in result.stdout
