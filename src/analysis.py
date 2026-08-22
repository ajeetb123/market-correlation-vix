"""Tests of the hypothesis: spikes in basket correlation lead spikes in the VIX."""

from __future__ import annotations

import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests


def rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    """How many trailing standard deviations above/below its own recent mean."""
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std()
    return ((series - mean) / std).dropna()


def cross_correlation(x: pd.Series, y: pd.Series, max_lag: int) -> pd.Series:
    """corr(x_t, y_{t+lag}) for lag in [-max_lag, max_lag].

    Positive lag: x today is compared against y `lag` days in the future.
    If the peak sits at a positive lag, x's moves tend to precede y's —
    i.e. average correlation leads the VIX, which is the hypothesis.
    """
    df = pd.concat([x, y], axis=1).dropna()
    x_s, y_s = df.iloc[:, 0], df.iloc[:, 1]
    out = {lag: x_s.corr(y_s.shift(-lag)) for lag in range(-max_lag, max_lag + 1)}
    return pd.Series(out).sort_index()


def identify_spike_events(z: pd.Series, threshold: float = 1.5, min_gap: int = 10) -> pd.DatetimeIndex:
    """Dates where the z-score first crosses above `threshold`.

    Crossings within `min_gap` trading days of a kept event are dropped, so
    one sustained spike counts once instead of once per day it stays elevated.
    """
    crossings = z.index[(z > threshold) & (z.shift(1) <= threshold)]
    if len(crossings) == 0:
        return crossings
    # Compare by position in z's trading-day index, not calendar days elapsed -
    # `(d - kept[-1]).days` would overcount the gap across a weekend/holiday
    # (e.g. Friday to the next Monday is 3 calendar days but 1 trading day).
    positions = z.index.get_indexer(crossings)
    kept = [0]
    for i in range(1, len(crossings)):
        if positions[i] - positions[kept[-1]] >= min_gap:
            kept.append(i)
    return crossings[kept]


def event_study(vix: pd.Series, events: pd.DatetimeIndex, horizon: int = 10) -> dict:
    """Compares VIX's forward `horizon`-day % change after correlation-spike
    events against the unconditional distribution of `horizon`-day VIX changes,
    via Welch's t-test (event sample is small and need not share the baseline's
    variance).
    """
    fwd_change = vix.pct_change(horizon).shift(-horizon)
    event_changes = fwd_change.reindex(events).dropna()
    # The event days themselves are excluded from the "unconditional" baseline -
    # otherwise the control group would quietly include the exact days it's
    # supposed to be compared against, biasing it toward the event group.
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
    """Tests whether lagged x (avg correlation) helps predict y (VIX) beyond
    y's own history. Returns {lag: p_value} from the SSR F-test at each lag;
    a small p-value at lag k is evidence x Granger-causes y at that lag.
    """
    df = pd.concat([y, x], axis=1).dropna()
    results = grangercausalitytests(df, maxlag=max_lag, verbose=False)
    return {lag: res[0]["ssr_ftest"][1] for lag, res in results.items()}
