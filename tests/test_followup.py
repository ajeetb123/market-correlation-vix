"""Tests for the follow-up study (synthetic data, no network)."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from vixagent import cli
from vixagent.analysis.followup import one_sided_p_negative, run_followup
from vixagent.analysis.frames import FrameStore
from vixagent.config import (
    FollowupConfig,
    FollowupPrereg,
    Settings,
    load_followup,
    load_followup_prereg,
    load_settings,
)
from vixagent.report.followup import render_followup_md, verdict_sentence
from vixagent.utils.jsonable import to_jsonable

PREREG = {
    "hypothesis": "The coefficient on z is negative.",
    "primary": {"window": 21, "horizon": 10, "include_controls": True, "alpha": 0.05},
    "note": "test",
}


def test_one_sided_p() -> None:
    assert one_sided_p_negative(-2.0, 0.04) == pytest.approx(0.02)
    assert one_sided_p_negative(2.0, 0.04) == pytest.approx(0.98)


def test_real_followup_config() -> None:
    s, cfg = load_followup(load_settings())
    assert len(cfg.universe) == 9 and "SPY" not in cfg.universe
    assert s.universe["risk"] == cfg.universe
    assert cfg.periods.holdout[1] < cfg.periods.exploration[0]
    assert cfg.periods.exploration[0] == load_settings().data.start


def test_missing_prereg_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_followup_prereg(tmp_path / "nope.yaml")


def test_run_followup_on_panel(panel: pd.DataFrame, small_settings: Settings) -> None:
    store = FrameStore(panel, small_settings)
    period = small_settings.period("full")
    r = run_followup(store, "risk", 21, 10, period, include_controls=True)
    assert set(r.regression.coefs) == {"const", "z", "vix_mom_5d", "log_vix"}
    assert r.regression.hac_maxlags == 10
    assert r.coef_z == r.regression.coefs["z"]["coef"]
    assert 0 <= r.p_one_sided <= 1
    base = run_followup(store, "risk", 21, 10, period, include_controls=False)
    assert set(base.regression.coefs) == {"const", "z"}


def _results(coef: float, p: float) -> dict[str, Any]:
    reg = {"n": 100, "r2": 0.1, "hac_maxlags": 10, "coefs": {}}
    row = {"period": ["1998-12-22", "2007-04-30"], "include_controls": True,
           "regression": reg, "coef_z": coef, "t_hac_z": -2.5, "p_one_sided": p}  # fmt: skip
    return {
        "universe": ["XLB"],
        "preregistered": PREREG,
        "holdout_primary": row,
        "holdout_secondary": row,
        "exploration_same_spec": row,
        "data_snapshot_end": "2026-06-30",
    }


def test_verdicts() -> None:
    assert verdict_sentence(_results(-0.02, 0.01)).endswith("Supported at alpha = 0.05.")
    assert verdict_sentence(_results(-0.02, 0.2)).endswith("Not supported at alpha = 0.05.")
    md = render_followup_md(_results(-0.02, 0.01))
    assert "## Verdict" in md and "holdout, preregistered spec" in md


@pytest.fixture
def wired(
    monkeypatch: pytest.MonkeyPatch, panel: pd.DataFrame, small_settings: Settings, tmp_path: Path
) -> None:
    s = small_settings
    cfg = FollowupConfig.model_validate(
        {
            "universe": s.universe["risk"],
            "data": s.data.model_dump(),
            "periods": {"holdout": s.periods.train, "exploration": s.periods.test},
        }
    )
    monkeypatch.setattr(cli, "load_followup", lambda base: (s, cfg))
    monkeypatch.setattr(cli, "load_or_fetch", lambda settings: panel)
    monkeypatch.setattr(cli, "find_project_root", lambda: tmp_path)


def _missing() -> FollowupPrereg:
    raise FileNotFoundError


def test_explore_never_writes_holdout(wired: None, tmp_path: Path) -> None:
    result = CliRunner().invoke(cli.app, ["followup"])
    assert result.exit_code == 0, result.stdout
    assert "Holdout (1998-2007) not computed." in result.stdout
    assert not (tmp_path / "reports" / "followup.json").exists()


def test_holdout_refused_without_prereg(
    wired: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "load_followup_prereg", _missing)
    result = CliRunner().invoke(cli.app, ["followup", "--holdout"])
    assert result.exit_code == 1
    assert "Freeze the follow-up" in result.stdout
    assert not (tmp_path / "reports" / "followup.json").exists()


def test_holdout_writes_reports(
    wired: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "load_followup_prereg", lambda: FollowupPrereg.model_validate(PREREG))
    result = CliRunner().invoke(cli.app, ["followup", "--holdout"])
    assert result.exit_code == 0, result.stdout
    data = json.loads((tmp_path / "reports" / "followup.json").read_text())
    json.dumps(to_jsonable(data), allow_nan=False)
    assert set(data) >= {"holdout_primary", "holdout_secondary", "exploration_same_spec"}
    assert (tmp_path / "reports" / "followup.md").read_text().startswith("# Follow-up")
