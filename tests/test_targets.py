"""Tests for forward-looking targets."""

import math

import numpy as np
import pandas as pd
import pytest

from vixagent.targets import fwd_any_within, fwd_log_vix_change


def test_fwd_log_vix_change_hand_example() -> None:
    out = fwd_log_vix_change(pd.Series([10.0, 20, 40]), 1)
    assert out.iloc[0] == pytest.approx(math.log(2))
    assert out.iloc[1] == pytest.approx(math.log(2))
    assert math.isnan(out.iloc[2])
    assert str(out.name).startswith("fwd_")


@pytest.mark.parametrize("h", [1, 5, 10])
def test_last_h_rows_nan(h: int) -> None:
    v = pd.Series(np.linspace(10, 30, 50))
    out = fwd_log_vix_change(v, h)
    assert out.iloc[-h:].isna().all()
    assert out.iloc[:-h].notna().all()


def test_fwd_any_within_windows() -> None:
    flags = pd.Series(False, index=range(30))
    flags[[8, 25]] = True
    out = fwd_any_within(flags, 5)
    expected = set(range(3, 8)) | set(range(20, 25))
    assert set(np.flatnonzero(out.to_numpy())) == expected
    assert str(out.name).startswith("fwd_")


def test_fwd_any_within_strictly_after() -> None:
    flags = pd.Series(False, index=range(10))
    flags[4] = True
    out = fwd_any_within(flags, 3)
    assert not out[4]
    assert out[3]


def test_fwd_any_within_nan_is_false() -> None:
    flags = pd.Series([None, None, True, None], dtype=object)
    out = fwd_any_within(flags, 1)
    assert out.tolist() == [False, True, False, False]
