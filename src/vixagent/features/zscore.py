"""Trailing (backward-looking) z-score."""

from __future__ import annotations

import pandas as pd


def trailing_zscore(x: pd.Series, lookback: int) -> pd.Series:
    """z_t = (x_t - mean(x_{t-L}..x_{t-1})) / std(x_{t-L}..x_{t-1}), ddof=1.

    Why shift(1): the baseline must exclude today so today's value is compared
    against history only. Why not full-sample stats: that leaks future data.
    A std at or below 1e-12 yields NaN (not inf).
    """
    past = x.shift(1)
    mu = past.rolling(lookback).mean()
    sd = past.rolling(lookback).std()
    return ((x - mu) / sd.where(sd > 1e-12)).rename(f"z_{x.name}")
