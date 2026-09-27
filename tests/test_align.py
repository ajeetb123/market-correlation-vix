"""Tests for group alignment."""

import numpy as np
import pandas as pd

from vixagent.config import Settings
from vixagent.data.align import align_group


def test_drops_nan_rows_and_orders_columns(panel: pd.DataFrame, small_settings: Settings) -> None:
    p = panel.copy()
    ticker = small_settings.universe["risk"][1]
    p.loc[p.index[[10, 50, 90]], ticker] = np.nan
    aligned, dropped = align_group(p, small_settings, "risk")
    assert dropped == 3
    assert not aligned.isna().any().any()
    assert list(aligned.columns) == [*small_settings.universe["risk"], "^VIX"]
    assert len(aligned) == len(p) - 3


def test_nan_in_other_group_ticker_ignored(panel: pd.DataFrame, small_settings: Settings) -> None:
    p = panel.copy()
    only_in_all = small_settings.universe["all"][-1]
    p.loc[p.index[10], only_in_all] = np.nan
    _, dropped = align_group(p, small_settings, "risk")
    assert dropped == 0


def test_rows_outside_period_excluded(panel: pd.DataFrame, small_settings: Settings) -> None:
    extra = panel.iloc[:3].copy()
    extra.index = extra.index - pd.Timedelta(days=365)
    p = pd.concat([extra, panel])
    aligned, dropped = align_group(p, small_settings, "all")
    assert aligned.index[0] == panel.index[0]
    assert dropped == 0
