"""Turns prices into returns and does the rolling correlation math."""

from __future__ import annotations

import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Turns prices into daily % changes.

    This matters more than it sounds like it should: if you correlate the
    raw prices instead, two totally unrelated stocks that both just went up
    for 10 years in a row will look "correlated" even though it has nothing
    to do with how they actually trade day-to-day - which is the thing this
    whole project cares about.
    """
    return prices.pct_change().dropna(how="all")


def rolling_correlation_matrices(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """All the rolling pairwise correlation matrices, one per day.

    `result.loc[date]` gives back the full n-by-n correlation matrix using
    that day's trailing `window`-day window.
    """
    return returns.rolling(window).corr()


def correlation_snapshot(returns: pd.DataFrame, window: int, as_of: pd.Timestamp) -> pd.DataFrame:
    """Just the one n-by-n correlation matrix for the window ending on `as_of`."""
    corr = returns.rolling(window).corr()
    return corr.loc[as_of]


def average_pairwise_correlation(returns: pd.DataFrame, window: int) -> pd.Series:
    """One number per day: the average correlation across every pair of
    stocks in the basket. This is the "how much is everything moving
    together today" series I'm testing against the VIX.

    Averaging every unique pair (not just each stock vs. the overall
    average) so no single stock can quietly dominate the number.
    """
    n = returns.shape[1]  # number of stocks in the basket
    corr = returns.rolling(window).corr()

    # add up each stock's row of correlations (n numbers, including its own
    # 1.0 on the diagonal) - require all n to actually be there, so a
    # window that hasn't fully "warmed up" yet gives NaN instead of a
    # number averaged over too few pairs
    row_sums = corr.sum(axis=1, min_count=n)
    # add up all n of those row-sums per day
    total = row_sums.groupby(level=0).sum(min_count=n)

    # total right now = n diagonal 1.0s + 2x every unique pair's correlation
    # (draw it out on paper for a 3x3 matrix if this doesn't make sense, it
    # took me a minute to convince myself this was right too)
    n_pairs = n * (n - 1)
    avg = (total - n) / n_pairs
    avg.name = "avg_pairwise_corr"
    avg = avg.dropna()
    return avg
