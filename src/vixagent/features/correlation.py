"""Rolling average pairwise correlation."""

from __future__ import annotations

from itertools import combinations

import pandas as pd


def avg_pairwise_corr(returns: pd.DataFrame, window: int) -> pd.Series:
    """Mean of all pairwise rolling Pearson correlations over the trailing
    `window` rows (inclusive of the current row). NaN until `window` returns
    exist, and NaN if any pair is undefined (zero variance).

    Why pairwise with itertools.combinations: it is equivalent to averaging the
    upper triangle of the rolling correlation matrix, but simpler to verify.
    """
    cols = list(returns.columns)
    if len(cols) < 2:
        raise ValueError("need at least 2 assets")
    pairs = [returns[a].rolling(window).corr(returns[b]) for a, b in combinations(cols, 2)]
    stacked = pd.concat(pairs, axis=1).clip(-1.0, 1.0)
    return stacked.mean(axis=1, skipna=False).rename(f"avg_corr_{window}")


def pairwise_corr_matrix(
    returns: pd.DataFrame, end_date: pd.Timestamp, window: int
) -> pd.DataFrame:
    """Correlation matrix of the last `window` rows ending at end_date
    (inclusive). Raises ValueError if fewer than `window` rows exist."""
    upto = returns.loc[:end_date]
    if len(upto) < window:
        raise ValueError(f"need {window} rows ending {end_date}, have {len(upto)}")
    return upto.iloc[-window:].corr()
