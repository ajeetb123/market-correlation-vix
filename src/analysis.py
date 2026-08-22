"""Tests of the hypothesis: spikes in basket correlation lead spikes in the VIX."""

from __future__ import annotations

import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests


def rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    """How many standard deviations away from its own recent average today's
    value is. Basically answers "is today weird compared to the last year".
    """
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std()
    return ((series - mean) / std).dropna()


def cross_correlation(x: pd.Series, y: pd.Series, max_lag: int) -> pd.Series:
    """Slides y back and forth against x and checks the correlation at every
    lag from -max_lag to +max_lag, to see which offset lines them up best.

    If the best lag is a positive number, that means x's moves tend to come
    BEFORE y's moves - i.e. avg correlation leads the VIX, which is the
    whole hypothesis I'm testing here.
    """
    df = pd.concat([x, y], axis=1).dropna()
    x_s = df.iloc[:, 0]
    y_s = df.iloc[:, 1]

    results = {}
    for lag in range(-max_lag, max_lag):
        shifted_y = y_s.shift(-lag)
        results[lag] = x_s.corr(shifted_y)

    out = pd.Series(results)
    out = out.sort_index()
    return out


def identify_spike_events(z: pd.Series, threshold: float = 1.5, min_gap: int = 10) -> pd.DatetimeIndex:
    """Finds the days where the z-score first jumps above `threshold`.

    If it stays elevated for a bunch of days in a row that should count as
    ONE spike, not a new spike every single day, so anything within
    `min_gap` trading days of a spike I already kept gets skipped.
    """
    crossings = z.index[(z > threshold) & (z.shift(1) <= threshold)]
    if len(crossings) == 0:
        return crossings
    # using position in the index (trading days) instead of real calendar
    # days - otherwise a Friday -> Monday gap counts as 3 days when it's
    # really just 1 trading day apart
    positions = z.index.get_indexer(crossings)
    kept = [0]
    for i in range(1, len(crossings)):
        if positions[i] - positions[kept[-1]] >= min_gap:
            kept.append(i)
    return crossings[kept]


def event_study(vix: pd.Series, events: pd.DatetimeIndex, horizon: int = 10) -> dict:
    """Compares what the VIX does in the `horizon` days after a correlation
    spike against what it normally does over any random `horizon`-day
    stretch, using a t-test that doesn't assume both groups have the same
    amount of spread (the spike group is a lot smaller).
    """
    fwd_change = vix.pct_change(horizon).shift(-horizon)
    event_changes = fwd_change.reindex(events).dropna()
    # gotta pull the event days OUT of the "normal" comparison group, or
    # else the comparison secretly includes the exact days it's supposed to
    # be compared against
    baseline_changes = fwd_change.drop(index=events, errors="ignore").dropna()
    t_stat, p_value = stats.ttest_ind(event_changes, baseline_changes, equal_var=False)
    return {
        "n_events": len(event_changes),
        "event_mean_change": event_changes.mean(),
        "baseline_mean_change": baseline_changes.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "event_changes": event_changes,
    }


def granger_causality(x: pd.Series, y: pd.Series, max_lag: int = 10) -> dict[int, float]:
    """Checks whether x's past (avg correlation) helps predict y (VIX) any
    better than just using y's own past history. Returns a p-value for each
    lag - a small p-value means x is probably helping predict y there.
    """
    df = pd.concat([y, x], axis=1).dropna()
    results = grangercausalitytests(df, maxlag=max_lag, verbose=False)
    return {lag: res[0]["ssr_ftest"][1] for lag, res in results.items()}
