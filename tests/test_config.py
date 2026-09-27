"""Tests for configuration loading and validation."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from vixagent.config import (
    Settings,
    find_project_root,
    load_preregistered,
    load_settings,
)


def _raw_settings() -> dict[str, Any]:
    with open(find_project_root() / "config" / "settings.yaml") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


def test_loads_real_config() -> None:
    s = load_settings()
    assert len(s.universe["risk"]) == 6
    assert s.periods.train[1] < s.periods.test[0]


def test_loads_preregistered() -> None:
    assert load_preregistered(load_settings()).primary.window == 21


def test_rejects_train_after_test() -> None:
    d = _raw_settings()
    d["periods"]["train"] = ["2007-05-01", "2020-01-01"]
    with pytest.raises(ValidationError):
        Settings.model_validate(d)


def test_rejects_small_group() -> None:
    d = _raw_settings()
    d["universe"]["risk"] = ["SPY", "QQQ", "IWM"]
    with pytest.raises(ValidationError):
        Settings.model_validate(d)


def test_rejects_duplicate_tickers() -> None:
    d = _raw_settings()
    d["universe"]["risk"] = ["SPY", "SPY", "QQQ", "IWM", "EFA", "EEM"]
    with pytest.raises(ValidationError):
        Settings.model_validate(d)


def test_rejects_empty_grid_list() -> None:
    d = _raw_settings()
    d["grid"]["horizons"] = []
    with pytest.raises(ValidationError):
        Settings.model_validate(d)


def test_rejects_prereg_not_in_grid(tmp_path: Path) -> None:
    s = load_settings()
    prereg = {
        "primary": {
            "group": "risk",
            "window": 30,
            "z_threshold": 2.0,
            "horizon": 10,
            "clean_only": True,
            "direction": "corr_to_vix",
            "alpha": 0.05,
        },
        "note": "test",
    }
    path = tmp_path / "prereg.yaml"
    path.write_text(yaml.safe_dump(prereg))
    with pytest.raises(ValueError, match="window"):
        load_preregistered(s, path)


def test_find_project_root() -> None:
    assert (find_project_root() / "pyproject.toml").is_file()


def test_period_lookup() -> None:
    s = load_settings()
    assert s.period("train") == s.periods.train
    assert s.period("test") == s.periods.test
    assert s.period("full") == (s.data.start, s.data.snapshot_end)
