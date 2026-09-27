"""Sanity checks on downloaded price data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vixagent.data.fetch import DataError


def validate_prices(prices: pd.DataFrame, vix_ticker: str) -> list[str]:
    """Raise DataError on hard problems; return warning strings for soft ones.

    Errors: index not monotonic increasing; duplicate index values; any
    non-NaN value <= 0; any non-NaN VIX value outside [5, 100].
    Warnings: any non-VIX column with |daily log return| > 0.5 (list ticker
    and date); any gap > 5 calendar days between consecutive index dates.

    Why errors vs warnings: non-positive prices or an impossible VIX mean the
    data is corrupt. Huge moves and gaps can be real (crashes, market
    closures), so they are surfaced for a human to check, not rejected.
    """
    if not prices.index.is_monotonic_increasing:
        raise DataError("Price index is not monotonic increasing")
    if prices.index.has_duplicates:
        raise DataError("Price index has duplicate dates")
    values = prices.to_numpy(dtype=float)
    if (values[~np.isnan(values)] <= 0).any():
        raise DataError("Prices contain non-positive values")
    if vix_ticker in prices.columns:
        vix = prices[vix_ticker].dropna()
        bad = vix[(vix < 5) | (vix > 100)]
        if not bad.empty:
            raise DataError(f"VIX outside [5, 100] on {bad.index[0].date()}: {bad.iloc[0]}")

    warnings: list[str] = []
    for ticker in prices.columns:
        if ticker == vix_ticker:
            continue
        rets = np.log(prices[ticker].dropna()).diff().abs()
        for when, value in rets[rets > 0.5].items():
            warnings.append(f"{ticker}: |log return| {value:.2f} on {pd.Timestamp(when).date()}")
    gaps = prices.index.to_series().diff().dt.days
    for when, days in gaps[gaps > 5].items():
        warnings.append(f"Gap of {int(days)} calendar days ending {pd.Timestamp(when).date()}")
    return warnings
