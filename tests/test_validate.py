"""Tests for price validation."""

import numpy as np
import pandas as pd
import pytest

from vixagent.data.fetch import DataError
from vixagent.data.validate import validate_prices


def test_clean_panel_has_no_warnings(panel: pd.DataFrame) -> None:
    assert validate_prices(panel, "^VIX") == []


def test_non_monotonic_raises(panel: pd.DataFrame) -> None:
    with pytest.raises(DataError, match="monotonic"):
        validate_prices(panel.iloc[::-1], "^VIX")


def test_duplicate_index_raises(panel: pd.DataFrame) -> None:
    dup = pd.concat([panel.iloc[:5], panel.iloc[4:5], panel.iloc[5:10]])
    with pytest.raises(DataError, match="duplicate"):
        validate_prices(dup, "^VIX")


def test_non_positive_price_raises(panel: pd.DataFrame) -> None:
    bad = panel.copy()
    bad.iloc[10, 0] = 0.0
    with pytest.raises(DataError, match="non-positive"):
        validate_prices(bad, "^VIX")


@pytest.mark.parametrize("level", [4.0, 120.0])
def test_vix_out_of_range_raises(panel: pd.DataFrame, level: float) -> None:
    bad = panel.copy()
    bad.loc[bad.index[20], "^VIX"] = level
    with pytest.raises(DataError, match="VIX"):
        validate_prices(bad, "^VIX")


def test_nan_is_ignored(panel: pd.DataFrame) -> None:
    p = panel.copy()
    p.iloc[5, 0] = np.nan
    assert validate_prices(p, "^VIX") == []


def test_large_jump_warns(panel: pd.DataFrame) -> None:
    p = panel.copy()
    p.iloc[100:, 2] *= 1.7
    warnings = validate_prices(p, "^VIX")
    assert any(p.columns[2] in w for w in warnings)


def test_gap_warns(panel: pd.DataFrame) -> None:
    p = panel.iloc[:50].copy()
    idx = p.index.to_list()
    idx[25:] = [d + pd.Timedelta(days=10) for d in idx[25:]]
    p.index = pd.DatetimeIndex(idx, name="date")
    warnings = validate_prices(p, "^VIX")
    assert any("Gap" in w for w in warnings)
