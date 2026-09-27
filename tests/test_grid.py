"""Tests for the parameter grid and overfitting check."""

from itertools import product

import pandas as pd
import pytest

from vixagent.analysis.event_study import EventStudyResult
from vixagent.analysis.frames import FrameStore
from vixagent.analysis.grid import GridRow, overfitting_check, run_grid
from vixagent.config import Preregistered, Settings, load_preregistered, load_settings


def _es(lift: float, n_events: int) -> EventStudyResult:
    return EventStudyResult(n_events, 0, 0.0, 0.1, lift, 0.5, 10, [], [], None)


def _row(window: int, z: float, h: int, train_lift: float, n: int = 10) -> GridRow:
    return GridRow("risk", window, z, h, _es(train_lift, n), _es(1.0, n))


@pytest.fixture
def prereg() -> Preregistered:
    return load_preregistered(load_settings())


def test_run_grid_order(panel: pd.DataFrame, small_settings: Settings) -> None:
    rows = run_grid(FrameStore(panel, small_settings), small_settings)
    g = small_settings.grid
    assert len(rows) == 2 * 2 * 2 * 3
    expected = list(product(g.groups, g.windows, g.z_thresholds, g.horizons))
    assert [(r.group, r.window, r.z_threshold, r.horizon) for r in rows] == expected


def test_overfitting_check_picks_best_with_enough_events(prereg: Preregistered) -> None:
    rows = [
        _row(21, 2.0, 10, 1.1),
        _row(21, 1.5, 5, 1.8),
        _row(63, 2.0, 20, 5.0, n=3),
        _row(63, 1.5, 5, 1.8),
    ]
    check = overfitting_check(rows, prereg)
    assert check.n_combinations == 4
    assert check.best_in_sample is rows[1]
    assert check.preregistered is rows[0]


def test_overfitting_check_none_when_no_candidates(prereg: Preregistered) -> None:
    rows = [_row(21, 2.0, 10, float("nan")), _row(21, 1.5, 5, 3.0, n=2)]
    assert overfitting_check(rows, prereg).best_in_sample is None


def test_overfitting_check_requires_prereg_row(prereg: Preregistered) -> None:
    with pytest.raises(ValueError):
        overfitting_check([_row(63, 1.5, 5, 1.2)], prereg)
