"""Tests for the frame store, event study, and permutation test."""

import math

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic import make_dates
from vixagent.analysis.event_study import EventStudyParams, event_study_core, run_event_study
from vixagent.analysis.frames import FrameStore
from vixagent.analysis.permutation import circular_shift_pvalue
from vixagent.config import Settings
from vixagent.periods import eligible_mask


def _flags(n: int, positions: list[int] | np.ndarray) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    out[np.asarray(positions, dtype=int)] = True
    return out


def _core(src: np.ndarray, tgt: np.ndarray, h: int, clean_only: bool, n_perm: int, seed: int = 0):
    n = len(src)
    dates = make_dates(n)
    eligible = eligible_mask(dates, (dates[0].date(), dates[-1].date()), h, np.ones(n, bool))
    return event_study_core(src, tgt, eligible, dates, h, 5, clean_only, n_perm, seed)


def test_frame_store_caches(panel: pd.DataFrame, small_settings: Settings) -> None:
    store = FrameStore(panel, small_settings)
    assert store.frame("risk", 21) is store.frame("risk", 21)
    assert store.aligned("risk") is store.aligned("risk")
    assert store.frame("risk", 21) is not store.frame("risk", 63)
    assert not store.frame("risk", 21).equals(store.frame("risk", 63))


def test_spec_hand_example() -> None:
    res = _core(_flags(30, [5, 15]), _flags(30, [8, 25]), 5, False, 100)
    assert res.n_events == 2
    assert res.n_hits == 1
    assert res.hit_rate == 0.5
    assert res.base_rate == pytest.approx(0.4)
    assert res.lift == pytest.approx(1.25)
    assert res.hit_flags == [True, False]
    assert len(res.event_dates) == 2


def test_clean_filter_removes_same_day_event() -> None:
    res = _core(_flags(30, [5, 15]), _flags(30, [8, 15, 25]), 5, True, 100)
    assert res.n_events == 1
    assert res.event_dates == [make_dates(30)[5].strftime("%Y-%m-%d")]


def test_planted_lead_is_detected() -> None:
    n = 3000
    rng = np.random.default_rng(7)
    targets = 50 + 70 * np.arange(40) + rng.integers(0, 30, size=40)
    assert np.diff(targets).min() >= 40
    sources = targets - rng.integers(3, 8, size=40)
    res = _core(_flags(n, sources), _flags(n, targets), 10, True, 1000)
    assert res.lift > 2
    assert res.p_value < 0.05


def test_no_relationship_type_one_error() -> None:
    n = 3000
    significant = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        src = _flags(n, rng.choice(n, size=30, replace=False))
        tgt = _flags(n, rng.choice(n, size=40, replace=False))
        res = _core(src, tgt, 10, True, 500, seed=seed)
        significant += res.p_value < 0.05
    assert significant / 20 <= 0.2


def test_pvalue_is_deterministic() -> None:
    src, tgt = _flags(300, [20, 90, 150, 220]), _flags(300, [25, 100, 180, 260])
    a = _core(src, tgt, 10, True, 200, seed=3)
    b = _core(src, tgt, 10, True, 200, seed=3)
    assert a.p_value == b.p_value


def test_few_events_warning() -> None:
    res = _core(_flags(30, [5, 15]), _flags(30, [8, 25]), 5, False, 50)
    assert res.warning == "insufficient events (<5)"


def test_enough_events_no_warning() -> None:
    src = _flags(200, [10, 40, 70, 100, 130])
    res = _core(src, _flags(200, [12, 45]), 5, False, 50)
    assert res.warning is None


def test_zero_events_is_nan() -> None:
    res = _core(np.zeros(30, bool), _flags(30, [8]), 5, False, 50)
    assert res.n_events == 0
    assert math.isnan(res.hit_rate)
    assert math.isnan(res.lift)
    assert math.isnan(res.p_value)


def test_permutation_short_series_is_nan() -> None:
    src = _flags(10, [2])
    assert math.isnan(circular_shift_pvalue(src, src, np.ones(10, bool), 0.5, 5, 10, 0))


def test_eligible_days_respect_horizon() -> None:
    res = _core(_flags(30, [5, 26]), _flags(30, [8]), 5, False, 20)
    assert res.n_events == 1


def test_run_event_study_detects_planted_lead(
    panel: pd.DataFrame, small_settings: Settings
) -> None:
    store = FrameStore(panel, small_settings)
    params = EventStudyParams("risk", 21, 2.0, 10, "full", True, "corr_to_vix")
    res = run_event_study(store, params, small_settings)
    assert res.n_permutations == small_settings.event_study.n_permutations
    assert res.n_events >= 1
    assert res.lift > 1


def test_run_event_study_reverse_direction(panel: pd.DataFrame, small_settings: Settings) -> None:
    store = FrameStore(panel, small_settings)
    params = EventStudyParams("risk", 21, 2.0, 10, "full", False, "vix_to_corr")
    res = run_event_study(store, params, small_settings)
    planted = {panel.index[p].strftime("%Y-%m-%d") for p in (300, 700, 1100)}
    assert set(res.event_dates) == planted
