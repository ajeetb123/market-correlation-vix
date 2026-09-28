"""Tests for the paper-replication analysis (synthetic data, no network)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tests.fixtures.synthetic import correlated_returns, make_dates
from vixagent import cli
from vixagent.analysis.paper import (
    align_event_dates,
    leadup_equity_corr,
    leadup_gold_equity_corr,
    leadup_test,
    vix_doubling_events,
)
from vixagent.config import DataConfig, PaperConfig, PaperPrereg, Settings
from vixagent.report.paper import primary_verdicts, render_paper_md

PREREG: dict[str, Any] = {
    "hypotheses": {"equity_styles": "high", "gold_equity": "low"},
    "event_definition": "VIX doubles its trailing low.",
    "lookback": 30,
    "multiple": 1.4,
    "cooldown": 60,
    "window": 20,
    "paper_years": [2011],
    "primary_event_set": "new",
    "n_permutations": 200,
    "alpha_per_test": 0.025,
    "note": "test",
}


def test_vix_doubling_events() -> None:
    v = pd.Series(20.0, index=make_dates(400))
    v.iloc[70:75] = 45.0
    v.iloc[100] = 45.0  # within the cooldown of the first episode
    v.iloc[300] = 45.0  # far later: a new event
    ev = vix_doubling_events(v, lookback=63, multiple=2.0, cooldown=126)
    assert list(np.flatnonzero(ev.to_numpy())) == [70, 300]


def test_vix_doubling_needs_full_lookback() -> None:
    v = pd.Series(20.0, index=make_dates(100))
    v.iloc[10] = 45.0
    assert not vix_doubling_events(v, lookback=63).any()


def _returns(n: int, k: int, corr: float, seed: int) -> pd.DataFrame:
    arr = correlated_returns(n, k, corr, seed)
    return pd.DataFrame(arr, index=make_dates(n), columns=[f"S{i}" for i in range(k)])


def test_leadup_equity_excludes_event_day() -> None:
    r = _returns(200, 4, 0.5, 1)
    stat = leadup_equity_corr(r, 63)
    iu = np.triu_indices(4, 1)
    for t in (80, 150):
        expected = np.corrcoef(r.iloc[t - 63 : t].to_numpy(), rowvar=False)[iu].mean()
        assert stat.iloc[t] == pytest.approx(expected, abs=1e-10)
    assert stat.iloc[:63].isna().all()


def test_leadup_gold_equity_negative() -> None:
    r = _returns(200, 4, 0.9, 2)
    gold = -r["S0"]
    stat = leadup_gold_equity_corr(r, gold, 40).dropna()
    assert (stat < -0.8).all()


def test_leadup_measures_have_no_lookahead() -> None:
    r = _returns(400, 4, 0.5, 3)
    gold = pd.Series(np.random.default_rng(4).standard_normal(400) * 0.01, index=r.index)
    full_eq, full_gd = leadup_equity_corr(r, 63), leadup_gold_equity_corr(r, gold, 63)
    for cut in (150, 250, 399):
        d = r.index[cut]
        pd.testing.assert_series_equal(full_eq.loc[:d], leadup_equity_corr(r.loc[:d], 63))
        pd.testing.assert_series_equal(
            full_gd.loc[:d], leadup_gold_equity_corr(r.loc[:d], gold.loc[:d], 63)
        )


def test_align_event_dates_rolls_forward() -> None:
    idx = make_dates(10)
    missing = pd.DatetimeIndex([idx[3] + pd.Timedelta(hours=1)])
    flags = align_event_dates(missing, idx)
    assert list(np.flatnonzero(flags.to_numpy())) == [4]


def _stat_with_bumps(positions: list[int], bump: float, seed: int = 0) -> pd.Series:
    idx = make_dates(3000)
    s = pd.Series(np.random.default_rng(seed).normal(0.5, 0.05, 3000), index=idx)
    for p in positions:
        s.iloc[p] += bump
    return s


def test_leadup_test_detects_planted_high() -> None:
    pos = list(range(200, 2800, 250))
    stat = _stat_with_bumps(pos, 0.2)
    events = pd.Series(False, index=stat.index)
    events.iloc[pos] = True
    t = leadup_test(stat, events, "high", 1000, 0, 63, "equity_styles", "new")
    assert t.p_value < 0.01
    assert t.mean_leadup > t.typical
    assert len(t.event_dates) == len(pos)


def test_leadup_test_direction_low_and_null() -> None:
    pos = list(range(200, 2800, 250))
    events = pd.Series(False, index=_stat_with_bumps([], 0).index)
    events.iloc[pos] = True
    low = leadup_test(_stat_with_bumps(pos, -0.2), events, "low", 1000, 0, 63, "g", "new")
    assert low.p_value < 0.01
    null = leadup_test(_stat_with_bumps([], 0, seed=5), events, "high", 1000, 0, 63, "g", "new")
    assert null.p_value > 0.05


def test_leadup_test_skips_undefined_days_and_is_deterministic() -> None:
    stat = _stat_with_bumps([500], 0.3)
    stat.iloc[:100] = np.nan
    events = pd.Series(False, index=stat.index)
    events.iloc[[50, 500]] = True
    a = leadup_test(stat, events, "high", 300, 7, 63, "e", "new")
    b = leadup_test(stat, events, "high", 300, 7, 63, "e", "new")
    assert a.event_dates == [stat.index[500].strftime("%Y-%m-%d")]
    assert a.p_value == b.p_value


def test_leadup_test_no_events_is_nan() -> None:
    stat = _stat_with_bumps([], 0)
    t = leadup_test(stat, pd.Series(False, index=stat.index), "high", 10, 0, 63, "e", "new")
    assert np.isnan(t.p_value) and np.isnan(t.mean_leadup)


def _results(p: float) -> dict[str, Any]:
    test = {"measure": "equity_styles", "direction": "high", "event_set": "new",
            "event_dates": ["2015-08-21"], "event_values": [0.9], "mean_leadup": 0.9,
            "typical": 0.8, "p_value": p, "n_permutations": 10}  # fmt: skip
    return {
        "equity_styles": ["A"],
        "gold": "G",
        "preregistered": PREREG,
        "tests": [test, {**test, "event_set": "paper"}],
        "events": [{"date": "2015-08-21", "set": "new", "equity_styles": 0.9, "gold_equity": None}],
        "data_snapshot_end": "2026-06-30",
    }


def test_render_and_verdicts() -> None:
    assert primary_verdicts(_results(0.01))[0].endswith("Supported at alpha = 0.025.")
    assert primary_verdicts(_results(0.2))[0].endswith("Not supported at alpha = 0.025.")
    assert len(primary_verdicts(_results(0.01))) == 1
    md = render_paper_md(_results(0.01))
    assert "## Per-event lead-up values" in md and "n/a" in md


def test_cli_refuses_without_prereg(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> PaperPrereg:
        raise FileNotFoundError

    monkeypatch.setattr(cli, "load_paper_prereg", missing)
    result = CliRunner().invoke(cli.app, ["paper"])
    assert result.exit_code == 1
    assert "Freeze the replication" in result.stdout


def test_cli_writes_reports(
    monkeypatch: pytest.MonkeyPatch, panel: pd.DataFrame, small_settings: Settings, tmp_path: Path
) -> None:
    tickers = small_settings.universe["all"]
    cfg = PaperConfig(
        equity_styles=tickers[:4],
        gold=tickers[4],
        data=DataConfig(**small_settings.data.model_dump()),
    )
    monkeypatch.setattr(cli, "load_paper_prereg", lambda: PaperPrereg.model_validate(PREREG))
    monkeypatch.setattr(cli, "load_paper", lambda base: (small_settings, cfg))
    monkeypatch.setattr(cli, "load_or_fetch", lambda settings: panel)
    monkeypatch.setattr(cli, "find_project_root", lambda: tmp_path)
    (tmp_path / "README.md").write_text("<!-- PAPER:START -->\n<!-- PAPER:END -->\n")
    result = CliRunner().invoke(cli.app, ["paper"])
    assert result.exit_code == 0, result.stdout
    data = json.loads((tmp_path / "reports" / "paper_replication.json").read_text())
    assert len(data["tests"]) == 6
    assert {t["event_set"] for t in data["tests"]} == {"new", "paper", "all"}
    assert "Details:" in (tmp_path / "README.md").read_text()
