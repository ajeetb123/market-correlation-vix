"""Rolling correlation analytics for the asset basket."""

from __future__ import annotations

import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily percent returns.

    Correlating on returns rather than raw prices matters: two unrelated
    stocks that have simply both trended upward for a decade would show a
    high price correlation with no connection to how they actually trade
    day to day, which is what this whole project is trying to measure.
    """
    return prices.pct_change().dropna(how="all")


def rolling_correlation_matrices(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling pairwise correlation matrices.

    Returns a long-form frame indexed by (date, ticker) with one column per
    ticker — `result.loc[date]` is the full n-by-n correlation matrix as of
    that date's `window`-day trailing window.
    """
    return returns.rolling(window).corr()


def correlation_snapshot(returns: pd.DataFrame, window: int, as_of: pd.Timestamp) -> pd.DataFrame:
    """The n-by-n correlation matrix for the `window`-day period ending `as_of`."""
    corr = returns.rolling(window).corr()
    return corr.loc[as_of]


def average_pairwise_correlation(returns: pd.DataFrame, window: int) -> pd.Series:
    """Daily rolling average of the basket's off-diagonal pairwise correlations.

    This single number is the "how much is the basket moving together today"
    series the VIX lead-lag hypothesis is tested against. It's the mean of
    every unique pair's correlation, not just correlation-with-the-average,
    so it isn't dominated by whichever ticker happens to track the mean best.
    """
    n = returns.shape[1]
    corr = returns.rolling(window).corr()

    # Sum each ticker's row of correlations (n values, including its own 1.0
    # diagonal entry); require all n to be present so a still-warming-up
    # window produces NaN instead of a value averaged over too few pairs.
    row_sums = corr.sum(axis=1, min_count=n)
    # Sum the n row-sums for each date, again requiring all n present.
    total = row_sums.groupby(level=0).sum(min_count=n)

    # total = n diagonal ones + 2x the sum of unique off-diagonal pairs.
    n_pairs = n * (n - 1)
    avg = (total - n) / n_pairs
    avg.name = "avg_pairwise_corr"
    return avg.dropna()
