"""Tests for spike detection and declustering."""

import numpy as np
import pandas as pd

from vixagent.spikes import backward_any, events_from_days, spike_days


def _flags(n: int, positions: list[int]) -> pd.Series:
    s = pd.Series(False, index=range(n))
    s[positions] = True
    return s


def test_spec_declustering_example() -> None:
    events = events_from_days(_flags(60, [10, 11, 12, 40, 45]), 20)
    assert list(np.flatnonzero(events.to_numpy())) == [10, 40]


def test_cooldown_zero_keeps_all_days() -> None:
    days = _flags(30, [3, 4, 20])
    assert events_from_days(days, 0).tolist() == days.tolist()


def test_cooldown_boundary() -> None:
    cooldown = 20
    after = events_from_days(_flags(60, [5, 5 + cooldown + 1]), cooldown)
    assert after[5 + cooldown + 1]
    within = events_from_days(_flags(60, [5, 5 + cooldown]), cooldown)
    assert not within[5 + cooldown]


def test_backward_any() -> None:
    out = backward_any(_flags(30, [10]), 5)
    assert out[10:16].all()
    assert not out[9]
    assert not out[16]


def test_nan_treated_as_false() -> None:
    x = pd.Series([np.nan, 3.0, np.nan, 1.0])
    assert spike_days(x, 2.0).tolist() == [False, True, False, False]
    days = pd.Series([np.nan, True, np.nan, True], dtype=object)
    assert events_from_days(days, 1).tolist() == [False, True, False, True]
    assert backward_any(days, 0).tolist() == [False, True, False, True]
